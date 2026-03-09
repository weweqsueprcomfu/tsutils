import os
from typing import Any

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import folder_paths


ENCODER_NAME = "resnet34"
DEFAULT_RESIDUAL_SCALE = 0.10

_MODEL_CACHE: dict[tuple[str, str], tuple[nn.Module, dict[str, Any]]] = {}


class ResidualUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        strength: float = 1.0,
        residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    ):
        residual = torch.tanh(self.net(x)) * float(residual_scale) * float(strength)
        pred = torch.clamp(x + residual, 0.0, 1.0)
        return pred, residual


class TSUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
                "residual_scale": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 2.0, "step": 0.01}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "TS_Nodes/Upscaler"

    def _ensure_batch(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError(f"Expected image with shape [H, W, C] or [B, H, W, C], got: {tuple(image.shape)}")
        return image

    def _resolve_model_path(self, model_name: str) -> str:
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        if model_path is None or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found in upscale_models: {model_name}")
        return model_path

    def _load_checkpoint_file(self, model_path: str, device: torch.device):
        ext = os.path.splitext(model_path)[1].lower()

        if ext == ".safetensors":
            try:
                from safetensors.torch import load_file
            except Exception as e:
                raise RuntimeError("To load .safetensors install safetensors: pip install safetensors") from e
            ckpt = load_file(model_path, device=str(device))
        else:
            ckpt = torch.load(model_path, map_location=device)

        return ckpt

    def _extract_state_dict(self, ckpt):
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                return ckpt["model_state_dict"], ckpt
            if "state_dict" in ckpt:
                return ckpt["state_dict"], ckpt

            tensor_values = [v for v in ckpt.values() if torch.is_tensor(v)]
            if len(tensor_values) == len(ckpt) and len(ckpt) > 0:
                return ckpt, {}

        raise ValueError("Unsupported checkpoint format. Expected model_state_dict/state_dict or a raw state dict.")

    def _get_model(self, model_name: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = self._resolve_model_path(model_name)
        cache_key = (model_path, str(device))

        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached[0], cached[1], device

        ckpt = self._load_checkpoint_file(model_path, device)
        state_dict, meta = self._extract_state_dict(ckpt)

        model = ResidualUNet().to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        _MODEL_CACHE[cache_key] = (model, meta)
        return model, meta, device

    def process(self, model_name, strength, residual_scale, image):
        image = self._ensure_batch(image)
        model, meta, device = self._get_model(model_name)

        outputs = []

        with torch.no_grad():
            for i in range(image.shape[0]):
                img_i = image[i : i + 1]
                x = img_i.movedim(-1, 1).to(device)

                pred, _ = model(
                    x,
                    strength=float(strength),
                    residual_scale=float(residual_scale),
                )

                pred = pred.movedim(1, -1).clamp(0.0, 1.0).cpu()
                outputs.append(pred)

                del x, pred
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        result = torch.cat(outputs, dim=0)

        if isinstance(meta, dict) and meta.get("config") is not None:
            print(f"[TSUpscaler] loaded config: {meta['config']}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "TSUpscaler": TSUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSUpscaler": "TS Upscaler",
}
