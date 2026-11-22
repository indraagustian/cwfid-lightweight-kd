import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficientvit_b0 import EfficientViT_B0_SOTA
from eval_checkpoints import FastSCNN_KDWrapper


def main():
    parser = argparse.ArgumentParser(description="Export trained KD student checkpoint to ONNX (opset 12)")
    parser.add_argument("--student", type=str, choices=["EfficientViT_B0", "Fast_SCNN"], required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output .onnx file path")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    model = EfficientViT_B0_SOTA(3) if args.student == "EfficientViT_B0" else FastSCNN_KDWrapper(3)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )
    print(f"Exported ONNX: {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
