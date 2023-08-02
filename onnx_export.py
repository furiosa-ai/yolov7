import argparse
import os
import sys
from pathlib import Path

import torch
import onnx
from onnx import shape_inference

root_path = Path(__file__).parent
INPUT_SHAPE = (640, 640)


def load_model(weights_path, device):
    yolo_path = root_path / "yolov7"
    sys.path.insert(0, str(yolo_path))
    from models.experimental import attempt_load

    model = attempt_load(weights_path, map_location=device)
    return model


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="./yolov7/yolov7.pt",
        help="Path to weight file (.pt or .pth)",
    )  # --weights
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./Result.onnx",
        help="Path of onnx file to generate",
    )  # --onnx_path
    parser.add_argument(
        "--opset_version",
        type=int,
        default=13,
        help="the ONNX version to export the model to",
    )
    parser.add_argument(
        "--model_input_name", type=str, default="images", help="the model's input name"
    )
    parser.add_argument(
        "--model_output_name",
        type=str,
        default="output",
        help="the model's output name",
    )
    args = parser.parse_args()
    return args


def main():
    args = build_argument_parser()

    weights = args.weights
    onnx_path = args.onnx_path
    opset = args.opset_version
    input_names = args.model_input_name
    output_names = args.model_output_name

    device = torch.device("cpu")

    model = load_model(weights, device)
    x = torch.zeros(1, 3, *INPUT_SHAPE).to(device)
    print("\nStarting ONNX export with onnx %s..." % onnx.__version__)
    torch.onnx.export(
        model,
        x,
        onnx_path,
        opset_version=opset,
        input_names=[input_names],
        output_names=[output_names],
    )

    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)


if __name__ == "__main__":
    main()
