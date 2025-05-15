Here’s a `README.md` file for your code. It includes an explanation of the code, how to use it, and how to set the `threshold` for the `is_overlapping` method as a command-line argument using `argparse`.

---

# YOLOv7 + TSM Inference Pipeline

This repository contains a Python script for performing object detection using YOLOv7 and action recognition using TSM (Temporal Shift Module). The pipeline processes video frames, detects objects, and classifies actions (e.g., "Hitting" or "Shaking") based on the detected objects.

## Features

- **Object Detection**: Uses YOLOv7 for detecting objects in video frames.
- **Action Recognition**: Uses TSM (Temporal Shift Module) for classifying actions based on detected objects.
- **Overlap Detection**: Checks if bounding boxes of class `0` overlap with bounding boxes of class `1`.
- **Real-Time Visualization**: Displays the results in an OpenCV window with annotations for actions.

## Requirements

- Prepared the Yolo pt file 
- Prepare the TSM .engine format (use tsm_transform_engine(tensorRT))


## Usage

Run the script with the following command:

```bash
python main.py --source <input_video_or_webcam> --weights yolov7.pt --load-engine-path tsm_engine/sahk_20012025.engine --iou-thres 0.45 --overlap-thres 0.1
```

!! reminded that evaluate the output by watching the video in ```/runs``` folder , dont use --view-img tag.
### Command-Line Arguments

| Argument              | Description                                                                 | Default Value                     |
|-----------------------|-----------------------------------------------------------------------------|-----------------------------------|
| `--source`            | Path to the input video file or webcam (e.g., `0` for webcam).              | `inference/images`                |
| `--weights`           | Path to the YOLOv7 model weights file.                                      | `yolov7.pt`                       |
| `--img-size`          | Inference size (height and width in pixels).                                | `640`                             |
| `--conf-thres`        | Confidence threshold for object detection.                                  | `0.25`                            |
| `--iou-thres`         | IOU threshold for Non-Maximum Suppression (NMS).                            | `0.45`                            |
| `--overlap-thres`     | Overlap threshold for bounding box overlap detection (used in `is_overlapping`). | `0.1`                             |
| `--device`            | Device to run the model on (e.g., `cuda:0` or `cpu`).                       | `''` (auto-detect)                |
| `--view-img`          | Display the results in an OpenCV window.                                    | `False`                           |
| `--save-txt`          | Save detection results to text files.                                       | `False`                           |
| `--save-conf`         | Save confidence scores in the output text files.                            | `False`                           |
| `--nosave`            | Do not save images or videos.                                               | `False`                           |
| `--classes`           | Filter detections by class (e.g., `--classes 0 1`).                         | `None` (all classes)              |
| `--agnostic-nms`      | Use class-agnostic NMS.                                                     | `False`                           |
| `--augment`           | Use augmented inference.                                                    | `False`                           |
| `--update`            | Update all models.                                                          | `False`                           |
| `--project`           | Directory to save results.                                                  | `runs/detect`                     |
| `--name`              | Experiment name.                                                            | `exp`                             |
| `--exist-ok`          | Allow overwriting existing project/name.                                    | `False`                           |
| `--no-trace`          | Disable model tracing.                                                      | `False`                           |
| `--load-engine-path`  | Path to the TSM engine file.                                                | `tsm_engine/sahk_20012025.engine` |

### Key Functions

1. **`detect()`**:
   - Main function for running object detection and action recognition.
   - Processes video frames, detects objects, and classifies actions.

2. **`split_detect()`**:
   - Filters detections for class `0` and checks if their bounding boxes overlap with class `1` bounding boxes.

3. **`is_overlapping()`**:
   - Checks if two bounding boxes overlap based on the Intersection over Union (IoU) metric.
   - The overlap threshold can be set via the `--overlap-thres` argument.

4. **`display_frames_with_annotations()`**:
   - Displays frames with annotations for actions (e.g., "Hitting", "Shaking", or "Normal").

5. **`process_video()`**:
   - Processes the input tensor for TSM inference.

6. **`get_animation()`**:
   - Generates animation tensors for detected objects.

   