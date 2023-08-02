# Yolov7 with Furiosa-SDK
This repository gives an example of optimally compiling and running a YOLOv7 model using the options provided by the [Furiosa SDK](https://furiosa-ai.github.io/docs/latest/ko/index.html).

## Setup
### Setup Environment
```sh
git clone git@github.com:furiosa-ai/yolov7.git
cd yolov7
git clone https://github.com/WongKinYiu/yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
conda create -n demo python=3.9
conda activate demo
pip install -r requirements.txt
```

## Export ONNX
Convert torch model to onnx model. 
```sh
python onnx_export.py --weights=./yolov7.pt --onnx_path=./yolov7.onnx --opset_version=13 --model_input_name=images --model_output_name=outputs
```

## Furiosa Quantization
Convert f32 onnx model to i8 onnx model using ```furiosa.quantinizer```. This involves a process for cutting off the post-processing elements.

```sh
python furiosa_quantize.py --onnx_path=./yolov7.onnx --dfg_path=./yolov7.dfg --opset_version=13 --calib_data=./images/train --model_input_name=images
```


```sh
# Argument
python furiosa_quantize.py -h
  --onnx_path ONNX_PATH
                        Path to onnx file
  --dfg_path DFG_PATH   Path to i8 onnx file
  --opset_version OPSET_VERSION
                        the ONNX version to export the model to
  --calib_data CALIB_DATA
                        Path to calibration data containing image files
  --calib_count CALIB_COUNT
                        How many images to use for calibration
  --model_input_name MODEL_INPUT_NAME
                        the model's input name
```

## Run
Create a session using the quantized model obtained from ```furiosa_quantize.py```. Use the sessions you create to make inferences on your test data set.

### Example
```sh
python furiosa_eval.py --dfg_path=./yolov7.dfg --eval_data_path=./images/test --output_path=./output
```

```sh
# Argument
python furiosa_eval.py -h
  --dfg_path DFG_PATH   Path to dfg file
  --eval_data_path EVAL_DATA_PATH
                        Path to evaluation data containing image files
  --eval_count EVAL_COUNT
                        How many images to use for evaluation
  --output_path OUTPUT_PATH
                        Path to result image
  ...
```
