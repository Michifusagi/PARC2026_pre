"""ポリシーサーバー（提出用テンプレート）

このファイルを編集して、自分のモデルを組み込んでください。
編集が必要なのは MyPolicy クラスの中身だけです。
それ以外のコード（サーバー部分、シリアライゼーション）は変更不可です。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --dry-run
"""

import argparse
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import msgpack
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from safetensors.torch import load_file

from minimal_smolvla.configuration import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from minimal_smolvla.modeling_smolvla import SmolVLAPolicy


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。以下のキーが含まれる:
                - "agentview_image": (128, 128, 3) uint8
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8
                - "robot0_joint_pos": (7,) float
                - "robot0_eef_pos": (3,) float
                - "robot0_eef_quat": (4,) float
                - "robot0_gripper_qpos": (2,) float

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
            seed: 推論の乱数シード。None の場合は乱数状態を変更しない。
        """
        ...


# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


class MyPolicy(BasePolicy):
    """自分のポリシーをここに実装する。

    例: チェックポイントをロードして推論する場合
        def __init__(self):
            self.model = torch.load("model_weights/checkpoint.pth")
            self.model.eval()

        def get_action(self, obs):
            image = obs["agentview_image"]
            # ... 前処理・推論 ...
            return action
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.instruction = ""

        model_dir = Path(__file__).resolve().parent / "model_weights"
        if not model_dir.exists():
            model_dir = Path(__file__).resolve().parent / "model_weights"
        if not model_dir.exists():
            raise FileNotFoundError("model_weights/ が見つかりません")

        self.model_dir = model_dir
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.policy = SmolVLAPolicy.from_pretrained(model_dir, device=str(self.device))
        self.tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        self.stats = load_file(
            str(model_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
            device=str(self.device),
        )

        self.front_image_size = self._image_size("observation.images.front")
        self.wrist_image_size = self._image_size("observation.images.wrist")
        self.debug_enabled = os.environ.get("PARC_DEBUG_POLICY", "0") == "1"
        self.debug_dir = Path(os.environ.get("PARC_DEBUG_DIR", "debug_policy"))
        self.debug_max_steps = int(os.environ.get("PARC_DEBUG_MAX_STEPS", "3"))
        self.debug_episode_index = -1
        self.debug_step_index = 0
        if self.debug_enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _image_size(self, feature_name: str) -> tuple[int, int]:
        feature = self.config.get("input_features", {}).get(feature_name, {})
        shape = feature.get("shape")
        if shape is None or len(shape) < 3:
            return (256, 256)
        return (int(shape[-2]), int(shape[-1]))

    def _debug_episode_dir(self) -> Path:
        episode_index = max(self.debug_episode_index, 0)
        path = self.debug_dir / f"episode_{episode_index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _debug_should_save_step(self) -> bool:
        return self.debug_enabled and self.debug_step_index < self.debug_max_steps

    @staticmethod
    def _image_to_uint8_hwc(image) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"デバッグ画像は3次元配列である必要があります: shape={array.shape}")
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _debug_save_image(self, name: str, image) -> None:
        if not self._debug_should_save_step():
            return
        from PIL import Image

        path = self._debug_episode_dir() / f"step_{self.debug_step_index:04d}_{name}.png"
        Image.fromarray(self._image_to_uint8_hwc(image)).save(path)

    def _debug_write_state(
        self,
        eef_pos: np.ndarray,
        eef_quat: np.ndarray,
        eef_axis_angle: np.ndarray,
        gripper_qpos: np.ndarray,
        state: np.ndarray,
        normalized_state: torch.Tensor,
    ) -> None:
        if not self._debug_should_save_step():
            return

        norm = normalized_state.detach().cpu().numpy().astype(np.float32, copy=False)
        record = {
            "step": self.debug_step_index,
            "instruction": self.instruction,
            "robot0_eef_pos": eef_pos.astype(float).tolist(),
            "robot0_eef_quat": eef_quat.astype(float).tolist(),
            "eef_axis_angle_xyzw": eef_axis_angle.astype(float).tolist(),
            "robot0_gripper_qpos": gripper_qpos.astype(float).tolist(),
            "observation_state_raw": state.astype(float).tolist(),
            "observation_state_normalized": norm.astype(float).tolist(),
            "normalized_state_stats": {
                "min": float(np.min(norm)),
                "max": float(np.max(norm)),
                "mean": float(np.mean(norm)),
                "std": float(np.std(norm)),
            },
        }
        path = self._debug_episode_dir() / "state_debug.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _image_tensor(self, image: np.ndarray, size: tuple[int, int], debug_name: str | None = None):
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"画像は3次元配列である必要があります: shape={array.shape}")
        if debug_name is not None:
            self._debug_save_image(f"{debug_name}_raw", array)
        if array.shape[0] == 3:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
        tensor = torch.flip(tensor, dims=(-2, -1))
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tuple(tensor.shape[-2:]) != tuple(size):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if debug_name is not None:
            self._debug_save_image(f"{debug_name}_processed", tensor)
        return tensor

    @staticmethod
    def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32).reshape(-1)
        if quat.shape != (4,):
            raise ValueError(f"robot0_eef_quat は4次元である必要があります: shape={quat.shape}")

        # LeRobot/LIBERO follows robosuite's xyzw quaternion convention.
        w = float(np.clip(quat[3], -1.0, 1.0))
        den = np.sqrt(1.0 - w * w)
        if np.isclose(den, 0.0):
            return np.zeros(3, dtype=np.float32)
        return (quat[:3] * (2.0 * np.arccos(w) / den)).astype(np.float32, copy=False)

    def _state_tensor(self, obs: dict[str, np.ndarray]):
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        if eef_pos.shape != (3,):
            raise ValueError(f"robot0_eef_pos は3次元である必要があります: shape={eef_pos.shape}")

        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
        eef_axis_angle = self._quat_to_axis_angle(eef_quat)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if gripper_qpos.shape != (2,):
            raise ValueError(f"robot0_gripper_qpos は2次元である必要があります: shape={gripper_qpos.shape}")

        state = np.concatenate([eef_pos, eef_axis_angle, gripper_qpos[:2]]).astype(
            np.float32,
            copy=False,
        )
        if state.shape != (8,):
            raise ValueError(f"observation.state は8次元である必要があります: shape={state.shape}")
        if state.dtype != np.float32:
            raise ValueError(f"observation.state はfloat32である必要があります: dtype={state.dtype}")
        tensor = torch.from_numpy(state).to(self.device)
        mean = self.stats["observation.state.mean"]
        std = self.stats["observation.state.std"]
        normalized = (tensor - mean) / (std + 1e-8)
        self._debug_write_state(
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            eef_axis_angle=eef_axis_angle,
            gripper_qpos=gripper_qpos,
            state=state,
            normalized_state=normalized,
        )
        return normalized

    def _tokenize(self) -> dict[str, torch.Tensor]:
        task = self.instruction
        if not task.endswith("\n"):
            task += "\n"
        tokens = self.tokenizer(
            [task],
            max_length=int(self.config.get("tokenizer_max_length", 48)),
            truncation=True,
            padding=self.config.get("pad_language_to", "max_length"),
            padding_side="right",
            return_tensors="pt",
        )
        return {
            OBS_LANGUAGE_TOKENS: tokens["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: tokens["attention_mask"].to(self.device, dtype=torch.bool),
        }

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        batch = {
            "observation.images.front": self._image_tensor(
                obs["agentview_image"],
                self.front_image_size,
                debug_name="agentview_image",
            ).unsqueeze(0).to(self.device),
            "observation.images.wrist": self._image_tensor(
                obs["robot0_eye_in_hand_image"],
                self.wrist_image_size,
                debug_name="robot0_eye_in_hand_image",
            ).unsqueeze(0).to(self.device),
            OBS_STATE: self._state_tensor(obs).unsqueeze(0),
        }
        batch.update(self._tokenize())

        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = action * (self.stats[f"{ACTION}.std"] + 1e-8) + self.stats[f"{ACTION}.mean"]

        action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7:
            raise ValueError(f"action は7次元である必要があります: shape={action.shape}")
        self.debug_step_index += 1
        return action

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        # instruction にはタスクの言語指示が渡される
        self.instruction = instruction
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.debug_episode_index += 1
        self.debug_step_index = 0
        if self.debug_enabled:
            (self._debug_episode_dir() / "state_debug.jsonl").write_text("", encoding="utf-8")
        if hasattr(self.policy, "reset"):
            self.policy.reset()


# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    seed = None
    if body:
        data = json.loads(body)
        instruction = data.get("instruction", "")
        seed = data.get("seed")
    _policy.reset(instruction=instruction, seed=seed)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
