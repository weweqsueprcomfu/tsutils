import os
import subprocess
import tempfile
import numpy as np
import torch


class TSDownscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (("soft", "tiktok", "hard"), {"default": "hard"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "TS_Nodes"

    TARGET_FPS = 30

    def _ensure_batch(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError(f"Expected image with shape [H, W, C] or [B, H, W, C], got {tuple(image.shape)}")
        return image

    def _mode_settings(self, mode: str) -> dict:
        presets = {
            "soft": {
                "crf": "20",
                "preset": "veryfast",
                "vf": "format=yuv420p",
            },
            "tiktok": {
                "crf": "27",
                "preset": "veryfast",
                "vf": "scale=iw*0.75:ih*0.75:flags=bicubic,scale=iw/0.75:ih/0.75:flags=bicubic,format=yuv420p",
            },
            "hard": {
                "crf": "32",
                "preset": "veryfast",
                "vf": "scale=iw*0.6:ih*0.6:flags=bilinear,scale=iw/0.6:ih/0.6:flags=bilinear,format=yuv420p",
            },
        }

        if mode not in presets:
            raise ValueError(f"Unknown mode: {mode}")
        return presets[mode]

    def _run_subprocess(self, cmd, input_bytes=None):
        try:
            result = subprocess.run(
                cmd,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return result
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not found. Install ffmpeg and make sure it is in PATH.") from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="ignore")
            stdout = e.stdout.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}") from e

    def _tensor_to_uint8(self, image: torch.Tensor) -> np.ndarray:
        image = self._ensure_batch(image).detach().cpu().clamp(0.0, 1.0)
        arr = (image.numpy() * 255.0).round().astype(np.uint8)
        return arr

    def _uint8_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32) / 255.0).clamp(0.0, 1.0)

    def _compress_frames_fast(self, image: torch.Tensor, mode: str) -> torch.Tensor:
        image = self._ensure_batch(image)
        frames = self._tensor_to_uint8(image)

        batch, height, width, channels = frames.shape
        if channels != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {channels}")

        settings = self._mode_settings(mode)

        with tempfile.TemporaryDirectory(prefix="ts_downscaler_") as tmp_dir:
            video_path = os.path.join(tmp_dir, "compressed.mp4")

            # 1. Кодируем raw RGB кадры в mp4
            encode_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(self.TARGET_FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                settings["preset"],
                "-crf",
                settings["crf"],
                "-vf",
                settings["vf"],
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                video_path,
            ]

            self._run_subprocess(encode_cmd, input_bytes=frames.tobytes())

            # 2. Декодируем обратно в raw RGB через stdout
            decode_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vsync",
                "0",
                "-vframes",
                str(batch),
                "-",
            ]

            result = self._run_subprocess(decode_cmd)
            raw = result.stdout

            expected_size = batch * height * width * 3
            if len(raw) != expected_size:
                raise RuntimeError(
                    f"Decoded frame size mismatch. Expected {expected_size} bytes, got {len(raw)} bytes."
                )

            out = np.frombuffer(raw, dtype=np.uint8).reshape(batch, height, width, 3)
            return self._uint8_to_tensor(out)

    def process(self, image, mode):
        result = self._compress_frames_fast(image, mode)
        return (result,)
