import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import EvalConfig, PerturbationConfig
from .environment import EnvironmentManager, TaskInfo
from .total_score import load_scoring_config

logger = logging.getLogger(__name__)


class PolicyInterface(Protocol):

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        ...


@dataclass
class EpisodeResult:
    task_name: str
    episode_id: int
    success: bool
    total_steps: int
    elapsed_time_sec: float

    joint_positions: list[np.ndarray] = field(default_factory=list)
    ee_positions: list[np.ndarray] = field(default_factory=list)
    ee_orientations: list[np.ndarray] = field(default_factory=list)
    gripper_qpos: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)


    collided: bool = False

    @property
    def trajectory(self) -> list[np.ndarray]:
        return self.joint_positions


@dataclass
class TaskResult:
    task_info: TaskInfo
    episodes: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    @property
    def avg_steps(self) -> float:
        successful = [e for e in self.episodes if e.success]
        if not successful:
            return 0.0
        return sum(e.total_steps for e in successful) / len(successful)

    @property
    def avg_time(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.elapsed_time_sec for e in self.episodes) / len(self.episodes)


class RolloutExecutor:

    def __init__(
        self,
        env_manager: EnvironmentManager,
        eval_config: EvalConfig,
        scoring_config: dict | None = None,
    ):
        self.env_manager = env_manager
        self.config = eval_config
        self._recorded_video = False


        self.scoring_config = scoring_config or load_scoring_config()

    def _should_record_episode(self) -> bool:
        return self.config.record_video_path is not None and not self._recorded_video

    def _image_from_obs(self, obs: dict[str, np.ndarray], key: str) -> np.ndarray | None:
        image = obs.get(key)
        if image is None:
            return None
        frame = np.asarray(image)
        if frame.ndim != 3:
            return None
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _video_frame(
        self,
        obs: dict[str, np.ndarray],
        step: int,
        reward: float,
        done: bool,
    ) -> np.ndarray | None:
        frames = []
        camera = self.config.record_video_camera
        if camera in ("both", "agentview"):
            frame = self._image_from_obs(obs, "agentview_image")
            if frame is not None:
                frames.append(frame)
        if camera in ("both", "wrist"):
            frame = self._image_from_obs(obs, "robot0_eye_in_hand_image")
            if frame is not None:
                frames.append(frame)
        if not frames:
            return None

        if len(frames) == 2 and frames[0].shape[0] != frames[1].shape[0]:
            import cv2

            height = frames[0].shape[0]
            width = round(frames[1].shape[1] * height / frames[1].shape[0])
            frames[1] = cv2.resize(frames[1], (width, height))
        rgb = frames[0] if len(frames) == 1 else np.concatenate(frames, axis=1)

        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        text = f"step={step} reward={reward:.3f} done={int(done)}"
        cv2.putText(
            bgr,
            text,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return bgr

    def _open_video_writer(self, frame: np.ndarray):
        import cv2

        path = self.config.record_video_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self.config.record_video_fps,
            (frame.shape[1], frame.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"動画 writer を開けませんでした: {path}")
        logger.info("デバッグ動画を保存します: %s", path)
        return writer

    def evaluate_task(
        self,
        policy: PolicyInterface,
        task_info: TaskInfo,
        perturbation: PerturbationConfig,
    ) -> TaskResult:
        logger.info(
            "タスク評価開始: %s (%d エピソード)",
            task_info.name, self.config.n_eval_episodes,
        )

        env = self.env_manager.create_env(task_info)


        init_states = self.env_manager.get_perturbed_init_states(
            task_info, perturbation, self.config.n_eval_episodes
        )


        collision_enabled = bool(self.scoring_config.get("collision", {}).get("enabled", True))
        obj_of_interest = (
            self.env_manager.get_obj_of_interest(task_info) if collision_enabled else set()
        )

        episodes: list[EpisodeResult] = []

        try:
            for ep_id in range(self.config.n_eval_episodes):
                result = self._run_episode(
                    env=env,
                    policy=policy,
                    task_info=task_info,
                    init_state=init_states[ep_id],
                    episode_id=ep_id,
                    perturbation=perturbation,
                    obj_of_interest=obj_of_interest,
                )
                episodes.append(result)

                if result.success:
                    logger.debug(
                        "  Episode %d: 成功 (%d steps)", ep_id, result.total_steps
                    )
                else:
                    logger.debug(
                        "  Episode %d: 失敗 (%d steps)", ep_id, result.total_steps
                    )
        finally:
            env.close()

        task_result = TaskResult(task_info=task_info, episodes=episodes)
        logger.info(
            "タスク評価完了: %s — 成功率 %.1f%% (平均 %.1f steps)",
            task_info.name, task_result.success_rate * 100, task_result.avg_steps,
        )
        return task_result

    def _run_episode(
        self,
        env: Any,
        policy: PolicyInterface,
        task_info: TaskInfo,
        init_state: np.ndarray,
        episode_id: int,
        perturbation: PerturbationConfig,
        obj_of_interest: set[str],
    ) -> EpisodeResult:
        start_time = time.time()
        joint_positions: list[np.ndarray] = []
        ee_positions: list[np.ndarray] = []
        ee_orientations: list[np.ndarray] = []
        gripper_qpos_log: list[np.ndarray] = []
        actions_log: list[np.ndarray] = []
        rewards_log: list[float] = []

        cc = self.scoring_config.get("collision", {})
        collision_enabled = bool(cc.get("enabled", True))
        collision_threshold = float(cc.get("threshold_m", 0.001))


        env.reset()
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()


        action_dim = env.robots[0].action_dim
        dummy_action = np.zeros(action_dim)
        for _ in range(10):
            obs, _, _, _ = env.step(dummy_action)


        object_init_pos: dict[str, np.ndarray] = {}
        if collision_enabled:
            object_init_pos = {
                k[:-4]: np.asarray(obs[k]).copy()
                for k in obs
                if k.endswith("_pos")
                and not k.startswith("robot0")
                and not k.endswith("_to_robot0_eef_pos")
                and k[:-4] not in obj_of_interest
            }
        object_max_disp: dict[str, float] = {}


        episode_seed = self.config.seed + episode_id
        policy.reset(instruction=task_info.language, seed=episode_seed)
        done = False
        total_steps = 0
        record_video = self._should_record_episode()
        video_writer = None

        try:
            for step in range(self.config.max_steps_per_episode):

                obs_for_policy = self.env_manager.apply_observation_noise(
                    obs, perturbation
                )


                action = policy.get_action(obs_for_policy)


                action = self.env_manager.apply_action_noise(action, perturbation)


                obs, reward, done, info = env.step(action)


                if record_video:
                    frame = self._video_frame(obs, step + 1, float(reward), bool(done))
                    if frame is not None:
                        if video_writer is None:
                            video_writer = self._open_video_writer(frame)
                        video_writer.write(frame)

                joint_positions.append(obs.get("robot0_joint_pos", np.zeros(7)).copy())
                ee_positions.append(obs.get("robot0_eef_pos", np.zeros(3)).copy())
                ee_orientations.append(obs.get("robot0_eef_quat", np.array([1, 0, 0, 0], dtype=np.float64)).copy())
                gripper_qpos_log.append(obs.get("robot0_gripper_qpos", np.zeros(2)).copy())
                actions_log.append(action.copy())
                rewards_log.append(float(reward))


                for name, p0 in object_init_pos.items():
                    cur = obs.get(name + "_pos")
                    if cur is not None:
                        d = float(np.sum(np.abs(np.asarray(cur) - p0)))
                        if d > object_max_disp.get(name, 0.0):
                            object_max_disp[name] = d

                total_steps = step + 1

                if total_steps % 50 == 0:
                    logger.info(
                        "  [進捗] %s: %d/%d steps (%.1fs)",
                        task_info.name, total_steps, self.config.max_steps_per_episode,
                        time.time() - start_time,
                    )

                if done:
                    break
        finally:
            if video_writer is not None:
                video_writer.release()
                logger.info("デバッグ動画を保存しました: %s", self.config.record_video_path)
            if record_video:
                self._recorded_video = True

        elapsed = time.time() - start_time


        collided = any(d > collision_threshold for d in object_max_disp.values())
        success = bool(done) and not collided

        return EpisodeResult(
            task_name=task_info.name,
            episode_id=episode_id,
            success=success,
            total_steps=total_steps,
            elapsed_time_sec=elapsed,
            joint_positions=joint_positions,
            ee_positions=ee_positions,
            ee_orientations=ee_orientations,
            gripper_qpos=gripper_qpos_log,
            actions=actions_log,
            rewards=rewards_log,
            collided=collided,
        )

    def evaluate_tasks(
        self,
        policy: PolicyInterface,
        task_infos: list[TaskInfo],
        perturbation: PerturbationConfig,
    ) -> list[TaskResult]:
        results = []
        for i, task_info in enumerate(task_infos):
            logger.info(
                "=== タスク %d/%d: %s ===",
                i + 1, len(task_infos), task_info.name,
            )
            result = self.evaluate_task(policy, task_info, perturbation)
            results.append(result)
        return results
