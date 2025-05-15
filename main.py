import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

win_width = 1440
win_height = 900

import pycuda.autoinit  # noqa
import pycuda.driver as cuda

import tensorrt as trt

BATCH_SIZE = 1
NUM_SEGMENTS = 8
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 3
SHIFT_DIV = 8

assert INPUT_H % 32 == 0 and INPUT_W % 32 == 0, \
    "Input height and width should be a multiple of 32."

EPS = 1e-5
INPUT_BLOB_NAME = "data"
OUTPUT_BLOB_NAME = "prob"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def print_config_info(opt):
    """
    Print the TSM configuration information based on the provided arguments

    Args:
        opt: Namespace object containing the configuration parameters
    """
    print("\n#################### TSM Configuration ####################")
    print(f"| {'Configuration':<25} | {'Value':<20} |")
    print("|---------------------------|----------------------|")
    print(f"| {'Engine Path':<25} | {opt.load_engine_path:<20} |")
    print(f"| {'Overlap Threshold':<25} | {opt.overlap_thres:<20} |")

    # Print flags with enabled/disabled status
    flags = [
        ('Upper Crop', opt.upper_crop),
        ('Overlap Check', opt.overlap),
        ('BBox Scaling', opt.scale),
        ('Save Video', opt.save_video)
    ]

    for name, flag in flags:
        status = "Enabled" if flag else "Disabled"
        print(f"| {name:<25} | {status:<20} |")

    # Only show scale factor if scaling is enabled
    if opt.scale:
        print(f"| {'Scale Factor':<25} | {opt.scale_factor:<20} |")

    # Only show output folder if video saving is enabled
    if opt.save_video:
        print(f"| {'Output Folder':<25} | {opt.output_folder:<20} |")

    print("########################################################\n")


import uuid


def save_tensor_as_mp4(tensor, output_folder, class_id, fps=8, input_color_format="BGR"):
    """
    Save a tensor of video frames as an MP4 file with a unique name including the class name.

    Args:
        tensor: NumPy array of shape (num_frames, height, width, channels) with values in [0, 255]
        output_folder: Directory to save the video
        class_id: Predicted class ID (1 for Hitting, 2 for Shaking)
        fps: Frames per second for the video (default: 8)
        input_color_format: Color format of the input tensor, either "RGB" or "BGR" (default: "BGR")
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Map class_id to class name
    class_names = {1: "Hitting", 2: "Shaking"}
    class_name = class_names.get(class_id, "Unknown")

    # Generate a unique video name with class name and UUID
    video_name = f"video_{class_name}_{uuid.uuid4().hex}.mp4"

    # Get tensor dimensions
    num_frames, height, width, _ = tensor.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    output_path = os.path.join(output_folder, video_name)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for i in range(num_frames):
        frame = tensor[i].astype(np.uint8)  # Ensure uint8 format
        if input_color_format == "BGR":
            # Convert BGR to RGB to ensure correct color representation, then to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # If input is RGB, convert directly to BGR for OpenCV
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    # Release the VideoWriter
    out.release()
    print(f"Video saved to {output_path}")

def show_frame_with_controls(frame, frame_num, total_frames, desired_size):
    """
    Display a single frame with navigation controls
    Args:
        frame: The frame to display (in RGB format)
        frame_num: Current frame number (1-based index)
        total_frames: Total number of frames
        desired_size: Target size of the frame
    Returns:
        bool: True if should continue, False if should quit
    """
    win_name = f"Frame {frame_num}/{total_frames} (Padded to {desired_size}x{desired_size})"

    # Convert to BGR for OpenCV display
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Show frame and wait for key press
    cv2.imshow(win_name, display_frame)
    key = cv2.waitKey(0) & 0xFF

    # Close current window
    cv2.destroyWindow(win_name)

    if key == ord('q'):  # Quit early
        return False
    elif key != ord('n'):  # If pressed key isn't 'n'
        print("Press 'n' for next frame or 'q' to quit")
        return show_frame_with_controls(frame, frame_num, total_frames, desired_size)  # Recursive retry

    return True


# execute_async removed  use execute_async_v3 instead
def do_inference(context, padded_frames):
    num_bindings = context.engine.num_io_tensors  # 26

    bindings = []
    host_buffers = []
    device_buffers = []

    for i in range(num_bindings):
        binding_name = context.engine.get_tensor_name(i)
        binding_shape = context.engine.get_tensor_shape(binding_name)
        binding_size = trt.volume(binding_shape)
        dtype = trt.nptype(context.engine.get_tensor_dtype(binding_name))
        host_buffer = cuda.pagelocked_empty(binding_size, dtype)

        device_buffer = cuda.mem_alloc(host_buffer.nbytes)

        host_buffers.append(host_buffer)
        device_buffers.append(device_buffer)
        bindings.append(int(device_buffer))

    # print(padded_frames)
    # Fill the input buffer with padded_frames
    np.copyto(host_buffers[0], padded_frames)

    stream = cuda.Stream()

    # Transfer input data to the GPU
    cuda.memcpy_htod_async(device_buffers[0], host_buffers[0], stream)

    # set address
    for i in range(num_bindings):
        binding_name = context.engine.get_tensor_name(i)
        context.set_tensor_address(binding_name, int(device_buffers[i]))

    context.execute_async_v3(stream_handle=stream.handle)

    # Transfer predictions from GPU to host
    for i in range(1, num_bindings):
        cuda.memcpy_dtoh_async(host_buffers[i], device_buffers[i], stream)

    # Synchronize the stream
    stream.synchronize()

    return host_buffers


def center_pad_images(image_list, desired_size):
    num_images = len(image_list)
    padded_images = np.zeros((num_images, desired_size, desired_size, 3), dtype=np.float32)
    for i, image in enumerate(image_list):
        height, width, _ = image.shape
        pad_height = (desired_size - height) // 2
        pad_width = (desired_size - width) // 2
        padded_images[i, pad_height:pad_height + height, pad_width:pad_width + width] = image
    return padded_images


def center_pad_image(image, desired_size):
    h, w, _ = image.shape
    # Calculate padding amounts
    w_pad = max((h - w) // 2, 0)
    h_pad = max((w - h) // 2, 0)
    # Pad the image to make it square
    padded_image = cv2.copyMakeBorder(
        image,
        top=h_pad,
        bottom=h_pad,
        left=w_pad,
        right=w_pad,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Padding with zeros (black pixels)
    )
    # Resize to desired_size x desired_size
    padded_image = cv2.resize(padded_image, (desired_size, desired_size), interpolation=cv2.INTER_LINEAR)
    return padded_image


def resize_and_pad_images(image_list, desired_height, desired_width):
    # Get the number of images

    num_images = len(image_list)

    # Create an array to store the resized and padded images
    resized_padded_images = np.zeros((num_images, desired_height, desired_width, 3), dtype=np.float32)

    # Resize and pad each image
    for i, image in enumerate(image_list):
        # Resize the image to the maximum size
        height, width, _ = image.shape
        scale_factor_h = desired_height / height
        scale_factor_w = desired_width / width
        scale_factor = min(scale_factor_h, scale_factor_w)
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        # Calculate the padding values
        pad_height = (desired_height - resized_image.shape[0]) // 2
        pad_width = (desired_width - resized_image.shape[1]) // 2

        # Paste the resized image onto the new blank image with padding
        resized_padded_images[i, pad_height:pad_height + resized_image.shape[0],
        pad_width:pad_width + resized_image.shape[1]] = resized_image

    return resized_padded_images


import cv2
import numpy as np
import os


import cv2
import numpy as np
import os

def process_video(input_tensor, NUM_SEGMENTS, INPUT_H, BATCH_SIZE, INPUT_W, context, show_frames=False, save_video=False, output_folder="output_videos"):
    """
    Process video frames with optional visualization and video saving.

    Args:
        input_tensor: Input tensor from YOLO based on bounding box
        NUM_SEGMENTS: Number of segments (frames)
        INPUT_H: Height of input tensor
        BATCH_SIZE: Batch size for inference
        INPUT_W: Width of input tensor
        context: TensorRT execution context
        show_frames: If True, displays each processed frame with controls
        save_video: If True, saves the input tensor as an MP4 video when class_id is 1 or 2
        output_folder: Directory to save the video (default: 'output_videos')

    Returns:
        class_id: Predicted class ID
        id_prob: Probability score for the predicted class
    """
    input_tensor_np = input_tensor.numpy()
    save_tensor = input_tensor_np


    desired_size = INPUT_H
    frames = []

    for i in range(input_tensor_np.shape[0]):
        frame = input_tensor_np[i]

        # Convert color space if needed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply center padding
        frame = center_pad_image(frame, desired_size)

        # Show frame if flag is set
        if show_frames:
            should_continue = show_frame_with_controls(
                frame=frame,
                frame_num=i + 1,
                total_frames=input_tensor_np.shape[0],
                desired_size=desired_size
            )
            if not should_continue:
                break

        # Convert to float32 and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    if show_frames:
        cv2.destroyAllWindows()

    # Stack frames into a numpy array
    frames = np.stack(frames, axis=0)  # Shape: [NUM_SEGMENTS, H, W, C]

    # Add batch dimension
    frames = np.expand_dims(frames, axis=0)  # Shape: [BATCH_SIZE, NUM_SEGMENTS, H, W, C]

    # Permute dimensions to (BATCH_SIZE, NUM_SEGMENTS, C, H, W)
    frames = np.transpose(frames, (0, 1, 4, 2, 3))

    # Flatten frames if required by the model's input
    frames_flatten = frames.ravel()

    # Ensure the input size matches the model's expected input size
    input_shape = BATCH_SIZE * NUM_SEGMENTS * 3 * INPUT_H * INPUT_W
    pad_size = input_shape - frames_flatten.size
    print(f'pad_size: {pad_size}')

    if pad_size > 0:
        # Pad the input tensor if necessary
        batch_padded = np.pad(frames_flatten, (0, pad_size), mode='constant')
        print(f'batch_padded shape: {batch_padded.shape}')
    else:
        # Truncate the input tensor if necessary
        batch_padded = frames_flatten[:input_shape]

    # Perform inference using the TensorRT context
    host_buffer = do_inference(context, batch_padded)

    # Reshape the output to [BATCH_SIZE, num_classes]
    batch_output = host_buffer[-1].reshape(BATCH_SIZE, -1)

    # Get the predicted class ID and score in one line
    class_id, id_prob = np.argmax(batch_output[0]), np.max(batch_output[0])

    # Save the tensor as an MP4 video if save_video is True and class_id is 1 or 2
    if save_video and class_id in [1, 2]:
        save_tensor_as_mp4(save_tensor, output_folder,class_id ,fps=8)

    return class_id, id_prob


import cv2


def display_frames_with_annotations(animation_dict, tsm_output_list, im0):
    """
    Display frames in an OpenCV window with annotations for class IDs 1 (Hitting) and 2 (Shaking).
    - If tsm_output_list is empty, display "Normal".
    - If tsm_output_list contains only 0s, display "Normal".
    - If tsm_output_list contains 1 or 2, display the corresponding labels.

    Args:
        animation_dict: Dictionary containing animation tensors for each object.
        tsm_output_list: List of class IDs (0 = Normal, 1 = Hitting, 2 = Shaking).
        im0: The original frame to overlay annotations on.
    """
    # Initialize a set to store unique labels
    unique_labels = set()

    # Check if the tsm_output_list is empty
    if not tsm_output_list:
        unique_labels.add("Normal")
    else:
        # Iterate through the tsm_output_list to find unique labels
        for class_id in tsm_output_list:
            if class_id == 1:
                unique_labels.add("Hitting")
            elif class_id == 2:
                unique_labels.add("Shaking")

        # If no Hitting or Shaking labels are found, add "Normal"
        if not unique_labels:
            unique_labels.add("Normal")

    # Overlay the unique labels on the frame
    y_offset = 30  # Vertical offset for each label
    for label in unique_labels:
        if label == "Hitting":
            color = (0, 0, 255)  # Red color for hitting
        elif label == "Shaking":
            color = (0, 255, 0)  # Green color for shaking
        elif label == "Normal":
            color = (0, 0, 0)  # White color for normal

        # Add the label to the frame with larger text
        cv2.putText(im0, label, (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        y_offset += 30  # Increment y_offset for the next label


def upper_crop(bbox, crop_ratio_thres=2.0):
    """
    Crop the bounding box to upper half if height is significantly larger than width.
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        crop_ratio_thres: threshold for height/width ratio to trigger cropping
    Returns:
        Cropped bounding box if ratio exceeds threshold, otherwise original bbox
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if height > crop_ratio_thres * width:
        # Crop to upper half
        new_height = height / 2
        return [x1, y1, x2, y1 + new_height]

    return bbox


def scale_up_bbox(bbox, scale_factor=2.0):
    """
    Scale up a bounding box by expanding it around its center point.

    Args:
        bbox: [x1, y1, x2, y2] coordinates of the bounding box
        scale_factor: Multiplier for scaling (default=2.0)

    Returns:
        Scaled bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate new dimensions
    new_width = width * scale_factor
    new_height = height * scale_factor

    # Calculate new coordinates
    new_x1 = max(0, center_x - (new_width / 2))
    new_y1 = max(0, center_y - (new_height / 2))
    new_x2 = center_x + (new_width / 2)
    new_y2 = center_y + (new_height / 2)

    return [new_x1, new_y1, new_x2, new_y2]


def split_detect(det, overlap_thres=None, crop_ratio_thres=2.0, up_cropflag=None,
                 overlap_flag=None, scale_up_flag=False, scale_factor=2.0):
    """
    Filter detections for class ID 0 and return bounding boxes and class IDs.
    - If overlap_flag=True, only store class 0 boxes that overlap with class 1.
    - If up_cropflag=True, crop tall boxes to upper half.
    - If overlap_flag=None, store all class 0 boxes (no overlap check).

    Args:
        det: List of detections (each detection is a tensor with [x1,y1,x2,y2,conf,class]).
        overlap_thres: IoU threshold for overlap check (None if not used).
        crop_ratio_thres: Height/width ratio threshold for cropping (default=2.0).
        up_cropflag: If True, crop tall boxes to upper half.
        overlap_flag: If True, enforce overlap check with class 1 boxes.
    Returns:
        bbox_dict: {index: [x1,y1,x2,y2]} of filtered boxes.
        class_dict: {index: class_id} of filtered boxes.
    """
    bbox_dict = {}
    class_dict = {}

    # Extract all bounding boxes for class 1 (if overlap check is needed)
    class1_boxes = []
    if overlap_flag:
        class1_boxes = [detection[:4].tolist() for detection in det if int(detection[-1].item()) == 1]

    for i, detection in enumerate(det):
        class_id = int(detection[-1].item())

        if class_id == 0:  # Only process class 0
            bbox = detection[:4].tolist()

            # Apply upper crop if needed
            if up_cropflag:
                bbox = upper_crop(bbox, crop_ratio_thres)

            # Case 1: No overlap check → store all class 0 boxes
            if not overlap_flag:
                bbox_dict[i] = bbox
                class_dict[i] = class_id

            # Case 2: Overlap check → only store if overlapping with class 1
            else:
                for class1_box in class1_boxes:
                    if is_overlapping(bbox, class1_box, overlap_thres):
                        bbox_dict[i] = bbox
                        class_dict[i] = class_id
                        break  # No need to check other class1 boxes

                        # Scale up bounding box if needed

            if scale_up_flag:
                bbox = scale_up_bbox(bbox, scale_factor)

    return bbox_dict, class_dict


def is_overlapping(bbox1, bbox2, threshold=0.1):
    """
    Check if two bounding boxes overlap or are near each other.
    Args:
        bbox1: Bounding box 1 [x1, y1, x2, y2].
        bbox2: Bounding box 2 [x1, y1, x2, y2].
        threshold: Overlap threshold (e.g., 0.1 for 10% overlap).
    Returns:
        True if the bounding boxes overlap or are near each other, False otherwise.
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection area
    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    intersection_area = x_overlap * y_overlap

    # Calculate areas of the bounding boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate IoU (Intersection over Union)
    iou = intersection_area / (area1 + area2 - intersection_area)

    # Check if IoU is greater than the threshold
    return iou > threshold


from collections import deque

# Initialize a buffer to store the past 8 frames
frame_buffer = deque(maxlen=8)


def get_animation(im0, bbox_dict):
    """
    Generate individual animations for each detected object in the current frame.
    Args:
        im0: Current frame (numpy array).
        bbox_dict: Dictionary containing bounding box coordinates for class ID 0.
    Returns:
        animation_dict: Dictionary where keys are object identifiers and values are animation tensors.
    """
    global frame_buffer

    # Add the current frame to the buffer
    frame_buffer.append(im0.copy())

    # If we don't have enough frames, return None
    if len(frame_buffer) < 8:
        return None

    # Initialize a dictionary to store animations for each object
    animation_dict = {}

    # Iterate through each bounding box in the current frame
    for obj_id, bbox in bbox_dict.items():
        x1, y1, x2, y2 = map(int, bbox)  # Convert coordinates to integers

        # Initialize a list to store cropped regions for this object
        cropped_frames = []

        # Iterate through the past 8 frames
        for frame in frame_buffer:
            # Crop the region of interest (ROI) from the frame
            cropped_frame = frame[y1:y2, x1:x2]

            # # Resize the cropped frame to a fixed size (optional, for consistency)
            # cropped_frame = cv2.resize(cropped_frame, (INPUT_W,INPUT_H))  # Resize to 224,224

            # Append the cropped frame to the list
            cropped_frames.append(cropped_frame)

        # Convert the list of cropped frames to a tensor
        animation_tensor = torch.tensor(cropped_frames)  # Shape: (8, H, W, 3)

        # Store the animation tensor in the dictionary
        animation_dict[obj_id] = animation_tensor

    return animation_dict


def draw_labeled_boxes(img, bbox_dict, obj_tsm_mapping, names, cls_list, conf_list, colors, save_img=False,
                       view_img=False):
    # Define action labels and colors
    ACTION_LABELS = {
        0: "Normal",
        1: "Hitting",
        2: "Shaking"
    }
    ACTION_COLORS = {
        0: (255, 255, 255),  # White for normal
        1: (0, 0, 255),  # Red for hitting
        2: (0, 255, 0)  # Green for shaking
    }

    # Draw object detection bounding boxes
    if save_img or view_img:
        for i, (xyxy, cls, conf) in enumerate(zip(bbox_dict.values(), cls_list, conf_list)):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)

    # Dictionary to store label positions to avoid overlap
    label_positions = {}

    # Add action labels with probabilities to existing bounding boxes
    for obj_id, bbox in bbox_dict.items():
        tsm_data = obj_tsm_mapping.get(obj_id, (0, 0.0))  # Default to (Normal, 0.0)
        tsm_class, tsm_prob = tsm_data

        # Only label actions of interest
        if tsm_class in [1, 2]:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            action_label = f"{ACTION_LABELS[tsm_class]}: {tsm_prob:.2f}"

            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(
                action_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Slightly smaller font
                2
            )

            # Default position above bounding box
            text_x = x1
            text_y = y1 - 5

            # Check for overlaps with existing labels
            for (ex_x, ex_y, ex_w, ex_h) in label_positions.values():
                if (text_x < ex_x + ex_w and
                        text_x + text_width > ex_x and
                        text_y < ex_y + ex_h and
                        text_y + text_height > ex_y):
                    # If overlap, move this label down
                    text_y = ex_y + ex_h + 5

            # Store this label's position
            label_positions[obj_id] = (text_x, text_y, text_width, text_height)

            # Draw the action label with background for better visibility
            cv2.rectangle(
                img,
                (text_x, text_y - text_height - 5),
                (text_x + text_width, text_y + 5),
                ACTION_COLORS[tsm_class],
                -1  # Filled rectangle
            )
            cv2.putText(
                img,
                action_label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),  # White text
                2
            )

    return img


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # tsm
    if opt.load_engine_path:
        # load from local file
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime
        with open(opt.load_engine_path, "rb") as f:
            engine = f.read()

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # ########################################################
                # Filter detections for class ID 0 and get bounding boxes
                bbox_dict, class_dict = split_detect(det, opt.overlap_thres, 2, opt.upper_crop, opt.overlap, opt.scale,
                                                     opt.scale_factor)
                o_img0 = im0
                # Generate the animation dictionary
                animation_dict = get_animation(o_img0, bbox_dict)

                if animation_dict is not None:
                    # Process each object's animation tensor
                    tsm_output_list = []
                    tsm_output_proab_list = []
                    obj_id_list = []  # To maintain order of processed objects

                    for obj_id, animation_tensor in animation_dict.items():
                        print(f"Animation Tensor for Object {obj_id} Shape:", animation_tensor.shape)
                        if opt.load_engine_path:
                            print("TSM engine load successfully.")

                            # Create TensorRT context
                            runtime = trt.Runtime(TRT_LOGGER)
                            deserialized_engine = runtime.deserialize_cuda_engine(engine)
                            assert deserialized_engine
                            context = deserialized_engine.create_execution_context()
                            assert context

                            # Process the animation tensor
                            tsm_output_class, tsm_output_proab = process_video(animation_tensor, NUM_SEGMENTS, INPUT_H,
                                                                               BATCH_SIZE, INPUT_W, context,
                                                                               opt.sh_frames, opt.save_video,opt.output_folder)

                            # Store both object ID and TSM output
                            obj_id_list.append(obj_id)
                            tsm_output_list.append(tsm_output_class)
                            tsm_output_proab_list.append(tsm_output_proab)

                    # Create mapping of object IDs to their TSM outputs
                    obj_tsm_mapping = dict(zip(obj_id_list, zip(tsm_output_list, tsm_output_proab_list)))

                    cls_list = class_dict.values()  # Class IDs from your detection
                    conf_list = [det[i][4] for i in range(len(det))]  # Confidence scores from detection

                    # Call draw_labeled_boxes to draw both object detection boxes and action labels
                    im0 = draw_labeled_boxes(
                        img=im0,
                        bbox_dict=bbox_dict,
                        obj_tsm_mapping=obj_tsm_mapping,
                        names=names,
                        cls_list=cls_list,
                        conf_list=conf_list,
                        colors=colors,
                        save_img=False,  # Set to True if you want to save the image
                        view_img=True  # Set to True to display the image
                    )

                    # Display global status (top-left corner)
                    # display_frames_with_annotations(animation_dict, tsm_output_list, im0)
                    # Print the final list of TSM outputs
                    print("TSM Output List:", obj_tsm_mapping)
                ########################################################
            # Print time (inference + NMS)

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.resizeWindow(str(p), win_width, win_height)  # Resize the window to 1280x720
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '--w', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    ######################## TSM configuration ################
    parser.add_argument("--sh_frames", action='store_true',
                        help="Showing the frames after pre-processing")
    parser.add_argument("--load-engine-path", type=str, default="tsm_engine/sahk_20012025.engine",
                        help="load tsm engine file path")
    parser.add_argument("--overlap-thres", type=float, default=0.0,
                        help="Overlap threshold for bounding box overlap detection")
    parser.add_argument("--upper_crop", action='store_true',
                        help="Upper crop flag")
    parser.add_argument("--overlap", action='store_true',
                        help="Overlap flag")
    parser.add_argument("--scale", action='store_true',
                        help="Scale flag")
    parser.add_argument("--save_video", action='store_true',
                        help="Save video flag")
    parser.add_argument("--scale_factor", type=float, default=0.0,
                        help="The number of the scaling if trigger the scale flag")
    parser.add_argument("--output_folder", type=str,
                        help="The folder of storing the animation what TSM been process.")
    opt = parser.parse_args()
    print_config_info(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
