#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
import pickle
import platform
import sys
import math
from numbers import Number
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
from ray import tune
from ray.tune import Callback
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.policy.policy import PolicySpec
from tensorboardX import SummaryWriter

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
from utils import PathUtils, TrainingUtils

ON_MAC = platform.system() == "Darwin"
save = PPO

train_batch_size = 60000 if not ON_MAC else 200  # Jan 32768
num_workers = 1 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 60 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)
scenario_name = "flocking"
# model_name = "MyFullyConnectedNetwork"
model_name = "GPPO"


def _flatten_loggable_metrics(data, prefix=""):
    metrics = {}
    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            metrics.update(_flatten_loggable_metrics(value, full_key))
        elif isinstance(value, bool):
            metrics[full_key] = int(value)
        elif isinstance(value, Number):
            if math.isfinite(float(value)):
                metrics[full_key] = value
    return metrics


def _collect_loggable_media(data, prefix=""):
    media = {}
    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            media.update(_collect_loggable_media(value, full_key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, str):
                    item_key = full_key if len(value) == 1 else f"{full_key}/{index}"
                    media[item_key] = item
        elif isinstance(value, str):
            media[full_key] = value
    return media


def _to_gymnasium_space(space):
    if isinstance(space, gym.spaces.Box):
        return space
    if hasattr(space, "spaces"):
        if isinstance(space, tuple) or space.__class__.__name__ == "Tuple":
            return gym.spaces.Tuple(tuple(_to_gymnasium_space(s) for s in space.spaces))
        if isinstance(space.spaces, dict):
            return gym.spaces.Dict(
                {key: _to_gymnasium_space(value) for key, value in space.spaces.items()}
            )
    if space.__class__.__name__ == "Box":
        return gym.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    if space.__class__.__name__ == "Discrete":
        return gym.spaces.Discrete(space.n)
    if space.__class__.__name__ == "MultiDiscrete":
        return gym.spaces.MultiDiscrete(space.nvec)
    raise TypeError(f"Unsupported space type: {type(space)!r}")


class TensorBoardLoggingCallback(Callback):
    def __init__(self, config):
        self.config = config
        self._writers = {}

    def on_trial_start(self, iteration, trials, trial, **info):
        logdir = Path(trial.local_path) / "custom_tensorboard"
        logdir.mkdir(parents=True, exist_ok=True)
        self._writers[trial] = SummaryWriter(log_dir=str(logdir))

    def on_trial_result(self, iteration, trials, trial, result, **info):
        writer = self._writers.get(trial)
        if writer is None:
            return
        metrics = _flatten_loggable_metrics(result)
        media = _collect_loggable_media(result)
        step = int(metrics.get("training_iteration", iteration))
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
        if media:
            for key, value in media.items():
                writer.add_text(key, value, step)
        writer.flush()

    def on_trial_complete(self, iteration, trials, trial, **info):
        writer = self._writers.pop(trial, None)
        if writer is not None:
            writer.close()

    def on_trial_error(self, iteration, trials, trial, **info):
        writer = self._writers.pop(trial, None)
        if writer is not None:
            writer.close()


def train(
    share_observations,
    centralised_critic,
    restore,
    heterogeneous,
    max_episode_steps,
    use_mlp,
    aggr,
    topology_type,
    add_agent_index,
    continuous_actions,
    seed,
    notes,
    share_action_value,
):
    checkpoint_rel_path = "ray_results/joint/HetGIPPO/MultiPPOTrainer_joint_654d9_00000_0_2022-08-23_17-26-52/checkpoint_001349/checkpoint-1349"
    checkpoint_path = PathUtils.scratch_dir / checkpoint_rel_path
    params_path = checkpoint_path.parent.parent / "params.pkl"

    if centralised_critic and not use_mlp:
        if share_observations:
            group_name = "GAPPO"
        else:
            group_name = "MAPPO"
    elif use_mlp:
        group_name = "CPPO"
    elif share_observations:
        group_name = "GPPO"
    else:
        group_name = "IPPO"

    group_name = f"{'Het' if heterogeneous else ''}{group_name}"

    if restore:
        with open(params_path, "rb") as f:
            config = pickle.load(f)

    trainer = MultiPPOTrainer
    trainer_name = "MultiPPOTrainer" if trainer is MultiPPOTrainer else "PPOTrainer"
    env_config = {
        "device": "cpu",
        "num_envs": num_envs_per_worker,
        "scenario_name": scenario_name,
        "continuous_actions": continuous_actions,
        "max_steps": max_episode_steps,
        # Env specific
        "scenario_config": {
            "n_agents": 4,
            "n_obstacles": 2,
            "dist_shaping_factor": 1,
            "collision_reward": -0.1,
        },
    }
    env = TrainingUtils.env_creator(env_config)
    observation_space = _to_gymnasium_space(env.observation_space)
    action_space = _to_gymnasium_space(env.action_space)
    run_config = {
        "seed": seed,
        "framework": "torch",
        "_disable_preprocessor_api": True,
        "env": scenario_name,
        "kl_coeff": 0,
        "kl_target": 0.01,
        "lambda": 0.9,
        "clip_param": 0.2,  # 0.3
        "vf_loss_coeff": 1,  # Jan 0.001
        "vf_clip_param": float("inf"),
        "entropy_coeff": 0,  # 0.01,
        "train_batch_size": train_batch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "sgd_minibatch_size": 4096 if not ON_MAC else 100,  # jan 2048
        "num_sgd_iter": 45,  # Jan 30
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus_per_worker": 0,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "lr": 5e-5,
        "gamma": 0.99,
        "use_gae": True,
        "use_critic": True,
        "grad_clip": 40,
        "batch_mode": "complete_episodes",
        "model": {
            "vf_share_layers": share_action_value,
            "_disable_preprocessor_api": True,
            "custom_model": model_name,
            "custom_action_dist": (
                "hom_multi_action" if trainer is MultiPPOTrainer else None
            ),
            "custom_model_config": {
                "activation_fn": "tanh",
                "share_observations": share_observations,
                "gnn_type": "MatPosConv",
                "centralised_critic": centralised_critic,
                "heterogeneous": heterogeneous,
                "use_beta": False,
                "aggr": aggr,
                "topology_type": topology_type,
                "use_mlp": use_mlp,
                "add_agent_index": add_agent_index,
                "pos_start": 0,
                "pos_dim": 2,
                "vel_start": 2,
                "vel_dim": 2,
                "share_action_value": share_action_value,
                "trainer": trainer_name,
            },
        },
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "default_policy": PolicySpec(
                    observation_space=observation_space,
                    action_space=action_space,
                )
            },
            "policy_mapping_fn": lambda *args, **kwargs: "default_policy",
        },
        "evaluation_interval": 30,
        "evaluation_duration": 1,
        "evaluation_num_workers": 1,
        "evaluation_parallel_to_training": False,
        "evaluation_config": {
            "num_envs_per_worker": 1,
            # "explore": False,
            "env_config": {
                "num_envs": 1,
            },
            "callbacks": make_multi_callbacks(
                [
                    TrainingUtils.RenderingCallbacks,
                    TrainingUtils.EvaluationCallbacks,
                    TrainingUtils.HeterogeneityMeasureCallbacks,
                ]
            ),
        },
        "callbacks": make_multi_callbacks(
            [
                TrainingUtils.EvaluationCallbacks,
            ]
        ),
    }
    tune_callbacks = [
        TensorBoardLoggingCallback(
            config=run_config,
        )
    ]
    tune.run(
        trainer,
        name=group_name if model_name.startswith("GPPO") else model_name,
        callbacks=tune_callbacks,
        storage_path=str(PathUtils.scratch_dir / "ray_results" / scenario_name),
        stop={"training_iteration": int(os.environ.get("HETGPPO_STOP_ITERS", "500"))},
        restore=str(checkpoint_path) if restore else None,
        config=run_config if not restore else config,
    )


if __name__ == "__main__":
    TrainingUtils.init_ray(scenario_name=scenario_name, local_mode=ON_MAC)
    for seed in [0]:
        train(
            seed=seed,
            restore=False,
            notes="",
            # Model important
            share_observations=True,
            heterogeneous=False,
            # Other model
            share_action_value=True,
            centralised_critic=False,
            use_mlp=False,
            add_agent_index=False,
            aggr="add",
            topology_type="full",
            # Env
            max_episode_steps=100,
            continuous_actions=True,
        )
