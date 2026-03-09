import os
import subprocess
import tempfile
import numpy as np
from PIL import Image
import torch


class TSDowаааnscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (("soft", "tiktok", "hard"), {"default": "hard"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "TS_Nodes"

    def _ensure_batch(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image

    def _tensor_to_frames(self, image):
        image = self._ensure_batch(image).cpu().clamp(0, 1)
        frames = []
        for i in range(image.shape[0]):
            frame = (image[i].numpy() * 255).astype(np.uint8)
            frames.append(frame)
        return frames

    def _frames_to_tensor(self, frames):
        arr = np.stack(frames).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    def _write_frames(self, frames, folder):
        os.makedirs(folder, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(folder, f"frame_{i:06d}.png"))

    def _run(self, cmd):
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _mode_settings(self, mode):
        presets = {
            "soft": {"crf": "20", "preset": "medium"},
            "tiktok": {"crf": "26", "preset": "slow"},
            "hard": {"crf": "32", "preset": "slow"},
        }
        return presets[mode]

    def _compress(self, frames, mode):
        settings = self._mode_settings(mode)

        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)

            self._write_frames(frames, frames_dir)

            video = os.path.join(tmp, "compressed.mp4")

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                "30",
                "-i",
                os.path.join(frames_dir, "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                settings["preset"],
                "-crf",
                settings["crf"],
                "-pix_fmt",
                "yuv420p",
                video,
            ]
            self._run(cmd)

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video,
                os.path.join(out_dir, "frame_%06d.png"),
            ]
            self._run(cmd)

            files = sorted(os.listdir(out_dir))
            out_frames = []
            for f in files:
                img = Image.open(os.path.join(out_dir, f)).convert("RGB")
                out_frames.append(np.array(img))

            return out_frames

    def process(self, image, mode):
        image = self._ensure_batch(image)

        frames = self._tensor_to_frames(image)
        degraded = self._compress(frames, mode)

        result = self._frames_to_tensor(degraded)
        return (result,)
