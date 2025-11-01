import os

import cv2
import torch

from main_of_3d.input_2 import depth_norm
from midas.midas.dpt_depth import DPTDepthModel
from midas.midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as transforms
import numpy as np

from midas.mobile.android.models.src.main.assets.run_tflite import output



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
class input_video:
    def __init__(self,video_path="C:/Users/hp/OneDrive/Videos/Captures",output_dir="C:/Users/hp/OneDrive/Videos/Captures/dir_for_frame"):
        os.makedirs(output_dir,exist_ok=True)
        cap=cv2.VideoCapture(video_path)
        i=0
        if cap == None:
            print("video not found")
        else:
            while True:
                ret,frame=cap.read()
                if not ret:
                    break
                cv2.imwrite(f"{output_dir}/frame_{i:05d}.jpg",frame)
                i+=1
                cap.release()
            print("red the video")
    def _depth_mapper(self):
        input_dir = "C:/Users/hp/OneDrive/Videos/Captures/dir_for_frames"
        output_dir = "C:/Users/hp/OneDrive/Videos/Captures/dir_for_depths"
        for name in sorted(os.listdir(input_dir)):
            img = cv2.imread(os.path.join(input_dir, name))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sample = transform({"image": img_rgb})["image"]
            sample = torch.from_numpy(sample).unsqueeze(0)
            with torch.no_grad():
                prediction = model.forward(sample)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=sample.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
            depth_normal = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, name), depth_normal)
            print("performed the depth mapping")

input_video()