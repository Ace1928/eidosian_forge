import argparse
import numpy as np
import os
import ray
from ray import air, tune
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.curriculum_capable_env import CurriculumCapableEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
def curriculum_fn(train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    new_task = int(np.log10(train_results['episode_reward_mean']) + 2.1)
    new_task = max(min(new_task, 5), 1)
    print(f'Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}\nR={train_results['episode_reward_mean']}\nSetting env to task={new_task}')
    return new_task