# Temporal Shift Module

TSM-R50 from "TSM: Temporal Shift Module for Efficient Video Understanding" <https://arxiv.org/abs/1811.08383>

TSM is a widely used Action Recognition model. This TensorRT implementation is tested with TensorRT 5.1 and TensorRT 7.2.

For the PyTorch implementation, you can refer to [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) or [mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module).

More details about the shift module(which is the core of TSM) could to [test_shift.py](./test_shift.py).

## Tutorial

+ An example could refer to [demo.sh](./demo.sh)
  + Requirements: Successfully installed `torch>=1.3.0, torchvision`

+ Step 1: Train/Download TSM-R50 checkpoints from [offical Github repo](https://github.com/mit-han-lab/temporal-shift-module) or [MMAction2](https://github.com/open-mmlab/mmaction2)
  + Supported settings: `num_segments`, `shift_div`, `num_classes`.
  + Fixed settings: `backbone`(ResNet50), `shift_place`(blockres), `temporal_pool`(False).

+ Step 2: Convert PyTorch checkpoints to TensorRT weights.

In ```gen_wts.py```,
```shell
python gen_wts.py /path/to/pytorch.pth --out-filename /weights/tensorrt.wts
```

+ Step 3(Optional): Test Python API.
  + Modify configs in `tsm_r50.py`.
  + Inference with `tsm_r50.py`.

```python
# Supported settings
BATCH_SIZE = 1
NUM_SEGMENTS = 8
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 400   # Depends of your output classes
SHIFT_DIV = 8
```
+ Step 4 (Building engine and test engine)
  + building engine by loading tensorrt weight and save the engine path
  + inference from input video and load the engine. 
```shell
usage: tsm_r50_without_midoutput.py [-h] [--tensorrt-weights TENSORRT_WEIGHTS] [--input-video INPUT_VIDEO] [--save-engine-path SAVE_ENGINE_PATH] [--load-engine-path LOAD_ENGINE_PATH] [--test-mmaction2] [--mmaction2-config MMACTION2_CONFIG] [--mmaction2-checkpoint MMACTION2_CHECKPOINT] [--test-cpp] [--cpp-result-path CPP_RESULT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --tensorrt-weights TENSORRT_WEIGHTS
                        Path to TensorRT weights, which is generated by gen_weights.py
  --input-video INPUT_VIDEO
                        Path to local video file , or accept mutiples input from folder.
  --save-engine-path SAVE_ENGINE_PATH
                        Save engine to local file
  --load-engine-path LOAD_ENGINE_PATH
                        Saved engine file path
```
