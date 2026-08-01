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
from abc import ABC, abstractmethod
from pathlib import Path

import msgpack
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response

try:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy
except ImportError:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


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
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
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
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.instruction = ""

        model_dir = Path(__file__).resolve().parent / "model_weights"
        if not model_dir.exists():
            model_dir = Path(__file__).resolve().parent / "model_weights"
        if not model_dir.exists():
            raise FileNotFoundError("model_weights/ が見つかりません")

        self.policy = SmolVLAPolicy.from_pretrained(model_dir).to(self.device)
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            model_dir,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
            },
        )

        self.front_image_size = self._image_size("observation.images.front")
        self.wrist_image_size = self._image_size("observation.images.wrist")

    def _image_size(self, feature_name: str) -> tuple[int, int]:
        features = getattr(self.policy.config, "input_features", {}) or {}
        feature = features.get(feature_name)
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if shape is None or len(shape) < 3:
            return (256, 256)
        return (int(shape[-2]), int(shape[-1]))

    def _image_tensor(self, image: np.ndarray, size: tuple[int, int]):
        torch = self.torch
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"画像は3次元配列である必要があります: shape={array.shape}")
        if array.shape[0] == 3:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tuple(tensor.shape[-2:]) != tuple(size):
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def _state_tensor(self, obs: dict[str, np.ndarray]):
        joint_pos = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        gripper = np.array([gripper_qpos.mean() if gripper_qpos.size else 0.0], dtype=np.float32)
        state = np.concatenate([joint_pos[:7], gripper]).astype(np.float32, copy=False)
        if state.shape != (8,):
            raise ValueError(f"observation.state は8次元である必要があります: shape={state.shape}")
        return self.torch.from_numpy(state)

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        frame = {
            "observation.images.front": self._image_tensor(
                obs["agentview_image"],
                self.front_image_size,
            ),
            "observation.images.wrist": self._image_tensor(
                obs["robot0_eye_in_hand_image"],
                self.wrist_image_size,
            ),
            "observation.state": self._state_tensor(obs),
            "task": self.instruction,
        }

        batch = self.preprocess(frame)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocess(action)

        if isinstance(action, dict):
            action = action["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 7:
            raise ValueError(f"action は7次元である必要があります: shape={action.shape}")
        return action

    def reset(self, instruction: str = "") -> None:
        # instruction にはタスクの言語指示が渡される
        self.instruction = instruction
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
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
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
