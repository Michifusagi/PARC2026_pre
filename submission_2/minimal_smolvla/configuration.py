from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTION = "action"
OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


@dataclass
class PolicyFeature:
    type: str
    shape: tuple[int, ...]


class SmolVLAConfig:
    def __init__(self, data: dict[str, Any], model_dir: Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = device

        for key, value in data.items():
            setattr(self, key, value)

        self.n_obs_steps = int(getattr(self, "n_obs_steps", 1))
        self.chunk_size = int(getattr(self, "chunk_size", 50))
        self.n_action_steps = int(getattr(self, "n_action_steps", self.chunk_size))
        self.max_state_dim = int(getattr(self, "max_state_dim", 32))
        self.max_action_dim = int(getattr(self, "max_action_dim", 32))
        self.tokenizer_max_length = int(getattr(self, "tokenizer_max_length", 48))
        self.num_steps = int(getattr(self, "num_steps", 10))
        self.empty_cameras = int(getattr(self, "empty_cameras", 0))
        self.num_expert_layers = int(getattr(self, "num_expert_layers", -1))
        self.num_vlm_layers = int(getattr(self, "num_vlm_layers", -1))
        self.self_attn_every_n_layers = int(getattr(self, "self_attn_every_n_layers", -1))
        self.expert_width_multiplier = float(getattr(self, "expert_width_multiplier", 0.5))
        self.min_period = float(getattr(self, "min_period", 0.004))
        self.max_period = float(getattr(self, "max_period", 4.0))

        self.use_cache = bool(getattr(self, "use_cache", True))
        self.freeze_vision_encoder = bool(getattr(self, "freeze_vision_encoder", True))
        self.train_expert_only = bool(getattr(self, "train_expert_only", True))
        self.train_state_proj = bool(getattr(self, "train_state_proj", True))
        self.load_vlm_weights = bool(getattr(self, "load_vlm_weights", False))
        self.add_image_special_tokens = bool(getattr(self, "add_image_special_tokens", False))
        self.adapt_to_pi_aloha = bool(getattr(self, "adapt_to_pi_aloha", False))
        self.compile_model = bool(getattr(self, "compile_model", False))
        self.rtc_config = getattr(self, "rtc_config", None)

        self.attention_mode = getattr(self, "attention_mode", "cross_attn")
        self.prefix_length = int(getattr(self, "prefix_length", -1))
        self.pad_language_to = getattr(self, "pad_language_to", "max_length")
        self.vlm_model_name = str(self.model_dir / "vlm_processor")

        self.input_features = self._features(getattr(self, "input_features", {}))
        self.output_features = self._features(getattr(self, "output_features", {}))
        self.normalization_mapping = getattr(
            self,
            "normalization_mapping",
            {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        )
        self.resize_imgs_with_padding = tuple(getattr(self, "resize_imgs_with_padding", (512, 512)))

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device: str) -> "SmolVLAConfig":
        model_dir = Path(model_dir)
        data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        return cls(data, model_dir=model_dir, device=device)

    @staticmethod
    def _features(data: dict[str, Any]) -> dict[str, PolicyFeature]:
        features: dict[str, PolicyFeature] = {}
        for name, value in data.items():
            features[name] = PolicyFeature(type=value["type"], shape=tuple(value["shape"]))
        return features

    def validate_features(self) -> None:
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {k: v for k, v in self.input_features.items() if v.type == "VISUAL"}

    @property
    def action_feature(self) -> PolicyFeature:
        return self.output_features[ACTION]


def get_safe_dtype(dtype, device_type: str):
    # CPU kernels used by torch/transformers are much happier in float32.
    if device_type == "cpu":
        return dtype if dtype is not None else None
    return dtype


def require_package(package_name: str, extra: str | None = None) -> None:
    try:
        __import__(package_name)
    except ImportError as exc:
        suffix = f" for the '{extra}' extra" if extra else ""
        raise ImportError(f"Package '{package_name}' is required{suffix}.") from exc


def populate_queues(queues, batch, exclude_keys=None):
    return queues
