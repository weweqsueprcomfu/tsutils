import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import folder_paths


def patch_basicsr_torchvision_import():
    old_text = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new_text = "from torchvision.transforms.functional import rgb_to_grayscale"

    for base in map(Path, sys.path):
        degradations_path = base / "basicsr" / "data" / "degradations.py"

        if degradations_path.exists():
            text = degradations_path.read_text(encoding="utf-8")
            if old_text in text:
                degradations_path.write_text(text.replace(old_text, new_text), encoding="utf-8")
            break


patch_basicsr_torchvision_import()

from basicsr.archs.swinir_arch import SwinIR


WINDOW_SIZE = 8


class TSDenoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "TS_Nodes/Video"

    def load_model(self, model_name):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this node.")

        model_path = folder_paths.get_full_path("upscale_models", model_name)

        if model_path is None:
            raise FileNotFoundError(f"Model not found: {model_name}")

        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )

        state = torch.load(model_path, map_location="cpu")

        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]
        elif "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state, strict=True)
        model.eval()
        model.to("cuda")

        return model

    def pad(self, x):

        _, _, h, w = x.size()

        pad_h = (WINDOW_SIZE - h % WINDOW_SIZE) % WINDOW_SIZE
        pad_w = (WINDOW_SIZE - w % WINDOW_SIZE) % WINDOW_SIZE

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return x, h, w

    def process(self, images, model_name, batch_size, strength):

        model = self.load_model(model_name)

        original = images.permute(0, 3, 1, 2).contiguous().to("cuda")

        x, h, w = self.pad(original.clone())

        out_batches = []

        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i : i + batch_size]
                y_batch = model(x_batch)
                y_batch = y_batch[:, :, :h, :w]
                out_batches.append(y_batch)

        denoised = torch.cat(out_batches, dim=0)

        # strength mix
        y = original * (1 - strength) + denoised * strength

        y = y.permute(0, 2, 3, 1).contiguous().clamp(0, 1)

        return (y.cpu(),)
