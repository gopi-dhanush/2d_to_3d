import os
import cv2
import torch
from midas.midas.dpt_depth import DPTDepthModel
from midas.midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as transforms
import numpy as np

# from midas.mobile.android.models.src.main.assets.run_tflite import output



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
    def __init__(self,video_path="C:/Users/hp/OneDrive/Videos/Captures/input.mp4",output_dir="C:/Users/hp/OneDrive/Videos/Captures/dir_for_frames"):
        os.makedirs(output_dir,exist_ok=True)
        cap=cv2.VideoCapture(video_path)
        i=0
        if not cap.isOpened():
            print("Video not found or could not be opened.")

        else:
            while True:
                ret,frame=cap.read()
                if not ret:
                    break
                cv2.imwrite(f"{output_dir}/frame_{i:05d}.jpg",frame)
                i+=1
            cap.release()
            print("red the video")
    def _depth(self):
        self.input_dir = "C:/Users/hp/OneDrive/Videos/Captures/dir_for_frames"
        self.output_dir = "C:/Users/hp/OneDrive/Videos/Captures/dir_for_depths"
        os.makedirs(self.output_dir, exist_ok=True)
        for name in sorted(os.listdir(self.input_dir)):
            img = cv2.imread(os.path.join(self.input_dir, name))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sample = transform({"image": img_rgb})["image"]
            sample = torch.from_numpy(sample).unsqueeze(0)
            with torch.no_grad():
                prediction = model.forward(sample)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
            depth_normal = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)

            cv2.imwrite(os.path.join(self.output_dir, name), depth_normal)
            print("performed the depth mapping")
    def _depth_mapper(self,shift=15):
        self.output_video_path="C:/Users/hp/OneDrive/Videos/Captures/2d_to_3d.mp4"
        frame_files = sorted(os.listdir(self.input_dir))
        depth_files = sorted(os.listdir(self.output_dir))

        if len(frame_files) != len(depth_files):
            raise ValueError(" Mismatch between frame and depth count.")

        # Initialize video writer
        first_frame = cv2.imread(os.path.join(self.input_dir, frame_files[0]))
        height, width = first_frame.shape[:2]
        out = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,  # FPS
            (width * 2, height)  # side-by-side stereo
        )

        print("🎥 Generating stereoscopic 3D video...")

        for f_name, d_name in zip(frame_files, depth_files):
            frame = cv2.imread(os.path.join(self.input_dir, f_name))
            depth = cv2.imread(os.path.join(self.output_dir, d_name), cv2.IMREAD_GRAYSCALE)
            depth_norm = depth.astype(np.float32) / 255.0

            # Generate small horizontal parallax shifts based on depth
            disparity = (1.0 - depth_norm) * shift  # near objects shift more

            # Create left and right images using pixel shifting
            map_x_left = np.tile(np.arange(width), (height, 1)).astype(np.float32) - disparity
            map_x_right = np.tile(np.arange(width), (height, 1)).astype(np.float32) + disparity
            map_y = np.tile(np.arange(height), (width, 1)).T.astype(np.float32)

            left = cv2.remap(frame, map_x_left, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            right = cv2.remap(frame, map_x_right, map_y, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

            # Combine side-by-side for stereoscopic effect
            stereo_frame = np.concatenate((left, right), axis=1)
            out.write(stereo_frame)

        out.release()
        print(f"✅ Stereoscopic 3D video saved to: {self.output_video_path}")

proj=input_video()
proj._depth()