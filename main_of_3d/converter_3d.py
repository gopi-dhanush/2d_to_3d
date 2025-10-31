import cv2
import torch
from midas.midas.dpt_depth import DPTDepthModel
from midas.midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as transforms
import numpy as np
# Path to weights (downloaded from MiDaS releases)
model_path = "c:/Users/hp/PycharmProjects/gopi/midas/weights/dpt_large-midas-2f21e586.pt"

# Load model
model = DPTDepthModel(
    path=model_path,
    backbone="vitl16_384",
    non_negative=True,
)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    Resize(
        384, 384,
        resize_target=None,
        keep_aspect_ratio=True,
        ensure_multiple_of=32,
        resize_method="minimal",
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet(),
])
#//////--/-/-/-/-/-/-//-/-//-///-/-/-
def estimate_depth(frame):
    img_input = transform({"image": frame})["image"]
    img_input = torch.from_numpy(img_input).unsqueeze(0)

    with torch.no_grad():
        prediction = model.forward(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    return prediction
#sgfuisdgfsdfd
