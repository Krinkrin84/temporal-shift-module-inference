import argparse
import os
import struct

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
import cv2
import torch

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


def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), f'Unable to load weight file {file}'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


# motified
def add_shift_module(network, input, input_shape, num_segments=8, shift_div=8):
    batch_size = input_shape[0]  # Assume input_shape includes batch dimension
    fold = input_shape[2] // shift_div  # Adjust for channel dimension (index 2 with batch)

    # left slice
    left_split = network.add_slice(input,
                                   start=(0, 1, 0, 0, 0),  # Include batch dim
                                   shape=(batch_size, num_segments - 1, fold, input_shape[3], input_shape[4]),
                                   # Adjust for explicit batch
                                   stride=(1, 1, 1, 1, 1))  # Include batch stride
    assert left_split
    left_split_shape = (batch_size, 1, fold, input_shape[3], input_shape[4])  # Include batch dim
    left_blank = network.add_constant(shape=left_split_shape,
                                      weights=np.zeros(left_split_shape, np.float32))
    assert left_blank
    left = network.add_concatenation([left_split.get_output(0), left_blank.get_output(0)])
    assert left
    left.axis = 1  # Concatenate along the segment axis

    # mid slice
    mid_split_shape = (batch_size, 1, fold, input_shape[3], input_shape[4])  # Include batch dim
    mid_blank = network.add_constant(shape=mid_split_shape,
                                     weights=np.zeros(mid_split_shape, np.float32))
    assert mid_blank
    mid_split = network.add_slice(input,
                                  start=(0, 0, fold, 0, 0),  # Adjust for explicit batch
                                  shape=(batch_size, num_segments - 1, fold,
                                         input_shape[3], input_shape[4]),  # Include batch dim
                                  stride=(1, 1, 1, 1, 1))
    assert mid_split
    mid = network.add_concatenation([mid_blank.get_output(0), mid_split.get_output(0)])
    assert mid
    mid.axis = 1  # Concatenate along the segment axis

    # right slice
    right = network.add_slice(input,
                              start=(0, 0, 2 * fold, 0, 0),  # Include batch dim
                              shape=(batch_size, num_segments, input_shape[2] - 2 * fold,
                                     input_shape[3], input_shape[4]),  # Include batch dim
                              stride=(1, 1, 1, 1, 1))

    # Concatenate left, mid, right
    output = network.add_concatenation([left.get_output(0), mid.get_output(0), right.get_output(0)])
    assert output
    output.axis = 2  # Concatenate along the channel axis
    print("shift output shape:", output.get_output(0).shape)
    return output


def add_batch_norm_2d(network, weight_map, input, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    return network.add_scale(input=input,
                             mode=trt.ScaleMode.CHANNEL,
                             shift=shift,
                             scale=scale)


def bottleneck(network, weight_map, input, in_channels, out_channels, stride,
               layer_name, input_shape):
    shift = add_shift_module(network, input, input_shape, NUM_SEGMENTS,
                             SHIFT_DIV)
    assert shift

    conv1 = network.add_convolution_nd(input=shift.get_output(0),
                                       num_output_maps=out_channels,
                                       kernel_shape=(1, 1),
                                       kernel=weight_map[layer_name +
                                                         "conv1.weight"],
                                       bias=trt.Weights())
    assert conv1

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0),
                            layer_name + "bn1", EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1

    conv2 = network.add_convolution_nd(input=relu1.get_output(0),
                                       num_output_maps=out_channels,
                                       kernel_shape=(3, 3),
                                       kernel=weight_map[layer_name +
                                                         "conv2.weight"],
                                       bias=trt.Weights())
    assert conv2
    conv2.stride_nd = (stride, stride)
    conv2.padding_nd = (1, 1)

    bn2 = add_batch_norm_2d(network, weight_map, conv2.get_output(0),
                            layer_name + "bn2", EPS)
    assert bn2

    relu2 = network.add_activation(bn2.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu2

    conv3 = network.add_convolution_nd(input=relu2.get_output(0),
                                       num_output_maps=out_channels * 4,
                                       kernel_shape=(1, 1),
                                       kernel=weight_map[layer_name +
                                                         "conv3.weight"],
                                       bias=trt.Weights())
    assert conv3

    bn3 = add_batch_norm_2d(network, weight_map, conv3.get_output(0),
                            layer_name + "bn3", EPS)
    assert bn3

    if stride != 1 or in_channels != 4 * out_channels:
        conv4 = network.add_convolution_nd(
            input=input,
            num_output_maps=out_channels * 4,
            kernel_shape=(1, 1),
            kernel=weight_map[layer_name + "downsample.0.weight"],
            bias=trt.Weights())
        assert conv4
        conv4.stride_nd = (stride, stride)

        bn4 = add_batch_norm_2d(network, weight_map, conv4.get_output(0),
                                layer_name + "downsample.1", EPS)
        assert bn4

        ew1 = network.add_elementwise(bn4.get_output(0), bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    else:
        ew1 = network.add_elementwise(input, bn3.get_output(0),
                                      trt.ElementWiseOperation.SUM)
    assert ew1

    relu3 = network.add_activation(ew1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu3

    return relu3


# motifited
def create_engine(maxBatchSize, builder, dt, weights):
    weight_map = load_weights(weights)
    network = builder.create_network()

    # Explicit batch dimension added to the input
    data = network.add_input(INPUT_BLOB_NAME, dt, (maxBatchSize, NUM_SEGMENTS, 3, INPUT_H, INPUT_W))
    assert data

    conv1 = network.add_convolution_nd(input=data,
                                       num_output_maps=64,
                                       kernel_shape=(7, 7),
                                       kernel=weight_map["conv1.weight"],
                                       bias=trt.Weights())
    assert conv1
    conv1.stride_nd = (2, 2)
    conv1.padding_nd = (3, 3)

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), "bn1",
                            EPS)
    assert bn1

    relu1 = network.add_activation(bn1.get_output(0),
                                   type=trt.ActivationType.RELU)
    assert relu1

    pool1 = network.add_pooling_nd(input=relu1.get_output(0),
                                   window_size=trt.DimsHW(3, 3),
                                   type=trt.PoolingType.MAX)
    assert pool1
    pool1.stride_nd = (2, 2)
    pool1.padding_nd = (1, 1)

    print("pool1.shape:", pool1.get_output(0).shape)  # (4, 8, 64, 56, 56)

    cur_height = INPUT_H // 4
    cur_width = INPUT_W // 4
    x = bottleneck(network, weight_map, pool1.get_output(0), 64, 64, 1,
                   "layer1.0.", (BATCH_SIZE, NUM_SEGMENTS, 64, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.1.", (BATCH_SIZE, NUM_SEGMENTS, 256, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 256, 64, 1,
                   "layer1.2.", (BATCH_SIZE, NUM_SEGMENTS, 256, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 256, 128, 2,
                   "layer2.0.", (BATCH_SIZE, NUM_SEGMENTS, 256, cur_height, cur_width))

    cur_height = INPUT_H // 8
    cur_width = INPUT_W // 8
    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.1.", (BATCH_SIZE, NUM_SEGMENTS, 512, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.2.", (BATCH_SIZE, NUM_SEGMENTS, 512, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 512, 128, 1,
                   "layer2.3.", (BATCH_SIZE, NUM_SEGMENTS, 512, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 512, 256, 2,
                   "layer3.0.", (BATCH_SIZE, NUM_SEGMENTS, 512, cur_height, cur_width))

    cur_height = INPUT_H // 16
    cur_width = INPUT_W // 16
    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.1.", (BATCH_SIZE, NUM_SEGMENTS, 1024, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.2.", (BATCH_SIZE, NUM_SEGMENTS, 1024, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.3.", (BATCH_SIZE, NUM_SEGMENTS, 1024, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.4.", (BATCH_SIZE, NUM_SEGMENTS, 1024, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 256, 1,
                   "layer3.5.", (BATCH_SIZE, NUM_SEGMENTS, 1024, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 1024, 512, 2,
                   "layer4.0.", (BATCH_SIZE, NUM_SEGMENTS, 1024, cur_height, cur_width))

    cur_height = INPUT_H // 32
    cur_width = INPUT_W // 32
    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.1.", (BATCH_SIZE, NUM_SEGMENTS, 2048, cur_height, cur_width))

    x = bottleneck(network, weight_map, x.get_output(0), 2048, 512, 1,
                   "layer4.2.", (BATCH_SIZE, NUM_SEGMENTS, 2048, cur_height, cur_width))

    pool2 = network.add_pooling_nd(x.get_output(0),
                                   window_size=trt.DimsHW(cur_height, cur_width),
                                   type=trt.PoolingType.AVERAGE)
    assert pool2
    pool2.stride_nd = (1, 1)

    print('pool2 output shape:', pool2.get_output(0).shape)
    print(pool2.get_output(0).dtype)  # (4, 8, 2048, 1, 1)

    # Assume weights and bias are loaded from the weight_map
    kernel = weight_map['fc.weight']  # (6144, 1)
    bias = weight_map['fc.bias']  # (3, 1)

    # Reshape kernel to [3, 2048] for matrix multiplication
    kernel = kernel.reshape((OUTPUT_SIZE, 2048))  # Correct the shape of the kernel

    # Convert numpy arrays to TensorRT weights
    kernel_weights = trt.Weights(kernel)
    bias_weights = trt.Weights(bias)

    # Create constant layers for kernel and bias (converting them to ITensor)
    kernel_tensor = network.add_constant(shape=(OUTPUT_SIZE, 2048), weights=kernel_weights).get_output(0)
    bias_tensor = network.add_constant(shape=(1, 1, OUTPUT_SIZE), weights=bias_weights).get_output(0)

    print('kernel_tensor shape:', kernel_tensor.shape)
    print('bias_tensor shape:', bias_tensor.shape)

    # Reshape pool2 output from [4, 8, 2048, 1, 1] to [32, 2048] (4 * 8, 2048)
    shuffle_layer_added = network.add_shuffle(pool2.get_output(0))
    shuffle_layer_added.reshape_dims = (BATCH_SIZE * NUM_SEGMENTS, pool2.get_output(0).shape[2])  # (32, 2048)

    print('shuffle_layer_added shape:', shuffle_layer_added.get_output(0).shape)
    # Perform matrix multiplication: [32, 2048] @ [3, 2048]T -> [32, 3]
    fc1 = network.add_matrix_multiply(shuffle_layer_added.get_output(0), trt.MatrixOperation.NONE,
                                      kernel_tensor, trt.MatrixOperation.TRANSPOSE)

    print('fc1 shape:', fc1.get_output(0).shape)  #

    shuffle_layer_added2 = network.add_shuffle(fc1.get_output(0))
    shuffle_layer_added2.reshape_dims = (BATCH_SIZE, NUM_SEGMENTS, OUTPUT_SIZE)  # (4, 8, 3)

    # Add bias tensor: [4, 8, 3] + [1, 1, 3] -> [4, 8, 3]
    fc1_with_bias = network.add_elementwise(shuffle_layer_added2.get_output(0), bias_tensor,
                                            trt.ElementWiseOperation.SUM)

    # Print the final output shape (should be [4, 8, 3])
    print('fc1 add bias shape: ', fc1_with_bias.get_output(0).shape)

    # fc1 = network.add_matrix_multiply(input=pool2.get_output(0),
    #                                   num_outputs=OUTPUT_SIZE,
    #                                   kernel=weight_map['fc.weight'],
    #                                   bias=weight_map['fc.bias'])
    assert fc1

    # reshape = network.add_shuffle(fc1_with_bias.get_output(0))
    # assert reshape
    # reshape.reshape_dims = (NUM_SEGMENTS, OUTPUT_SIZE)

    reduce = network.add_reduce(fc1_with_bias.get_output(0),
                                op=trt.ReduceOperation.AVG,
                                axes=2,
                                keep_dims=False)
    assert reduce

    print('reduce shape:', reduce.get_output(0).shape)

    softmax = network.add_softmax(reduce.get_output(0))
    assert softmax
    softmax.axes = 2

    print('softmax shape:', softmax.get_output(0).shape)

    softmax.get_output(0).name = OUTPUT_BLOB_NAME
    network.mark_output(softmax.get_output(0))

    # Build engine
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 50)
    engine = builder.build_serialized_network(network, config)

    del network
    del weight_map

    return engine


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
    import cv2
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


def process_video(input_video, NUM_SEGMENTS, INPUT_H, BATCH_SIZE, INPUT_W, context):
    vid = input_video.lower().endswith('.mp4')
    if input_video and vid:
        cap = cv2.VideoCapture(input_video)
        sample_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        avg_interval = sample_length / float(NUM_SEGMENTS)
        base_offsets = np.arange(NUM_SEGMENTS) * avg_interval
        clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)

        desired_size = INPUT_H  # Assuming INPUT_H == INPUT_W == crop_size

        frames = []

        print('clip_offsets:', clip_offsets)
        # Read frames
        for i in range(max(clip_offsets) + 1):
            flag, frame = cap.read()
            if i in clip_offsets:
                # frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply center padding and resize
                frame = center_pad_image(frame, desired_size)
                # Convert to float32 and divide by 255 to normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                # Append the processed frame to the frames list
                frames.append(frame)

        frames = np.stack([frames], axis=0)

        frames = np.array(frames)  # Shape: (BATCH_SIZE, NUM_SEGMENTS, H, W, C)
        print('frames shape:', frames.shape)

        # Permute dimensions to (BATCH_SIZE, NUM_SEGMENTS, C, H, W)
        frames = np.transpose(frames, (0, 1, 4, 2, 3))

        # Flatten frames if required by the model's input
        frames_flatten = frames.ravel()

        # Ensure the input size matches the model's expected input size
        input_shape = BATCH_SIZE * NUM_SEGMENTS * 3 * INPUT_H * INPUT_W
        pad_size = input_shape - frames_flatten.size
        print('pad_size:', pad_size)
        if pad_size > 0:
            batch_padded = np.pad(frames_flatten, (0, pad_size), mode='constant')
            print('batch_padded shape:', batch_padded.shape)
        else:
            batch_padded = frames_flatten[:input_shape]

        # Proceed with inference
        host_buffer = do_inference(context, batch_padded)
        batch_output = (host_buffer[-1].reshape(BATCH_SIZE, -1))
        class_id = np.argmax(batch_output[0])
        # print model output
        print(batch_output.shape)
        print(batch_output[0])

        print(f'Result class id {class_id}, score {batch_output[0][class_id]:.2f}')


def main(args):
    assert not (args.save_engine_path and args.load_engine_path)

    if args.load_engine_path:
        # load from local file
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime
        with open(args.load_engine_path, "rb") as f:
            engine = f.read()
    else:
        # Create network and engine
        assert args.tensorrt_weights
        builder = trt.Builder(TRT_LOGGER)
        engine = create_engine(BATCH_SIZE, builder, trt.float32,
                               args.tensorrt_weights)
    assert engine
    # assert engine.num_bindings == 2

    if args.save_engine_path is not None:
        # save engine to local file
        with open(args.save_engine_path, "wb") as f:
            f.write(engine)
        print(f"{args.save_engine_path} Generated successfully.")

    # Create a runtime object to deserialize the engine
    runtime = trt.Runtime(TRT_LOGGER)

    # Deserialize the engine from serialized data
    deserialized_engine = runtime.deserialize_cuda_engine(engine)
    assert engine

    context = deserialized_engine.create_execution_context()
    assert context

    input_shape = BATCH_SIZE * NUM_SEGMENTS * 3 * INPUT_H * INPUT_W
    if not args.save_engine_path:s
        if os.path.isfile(args.input_video) and args.input_video.lower().endswith('.mp4'):
            process_video(args.input_video, NUM_SEGMENTS, INPUT_H, BATCH_SIZE, INPUT_W, context)
        else:
            print(f"Processing directory: {args.input_video}")
            for vid_file in os.listdir(args.input_video):
                print("**********************")
                print(vid_file)
                vid_path = os.path.join(args.input_video, vid_file)
                if os.path.isfile(vid_path) and vid_path.lower().endswith('.mp4'):
                    process_video(vid_path, NUM_SEGMENTS, INPUT_H, BATCH_SIZE, INPUT_W, context)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorrt-weights",
                        type=str,
                        default=None,
                        help="Path to TensorRT weights, which is generated by gen_weights.py")
    parser.add_argument("--input-video",
                        type=str,
                        default=None,
                        help="Path to local video file or a folder with videos")
    parser.add_argument("--save-engine-path",
                        type=str,
                        default=None,
                        help="Save engine to local file")
    parser.add_argument("--load-engine-path",
                        type=str,
                        default=None,
                        help="Saved engine file path")
    parser.add_argument("--video-folder,--f",
                        type=str,
                        default=None,
                        help="Read video from a folder.")

    main(parser.parse_args())
