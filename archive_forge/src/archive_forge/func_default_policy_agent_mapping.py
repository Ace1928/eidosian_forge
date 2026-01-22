import collections
import copy
import gymnasium as gym
import json
import os
from pathlib import Path
import shelve
import typer
import ray
import ray.cloudpickle as cloudpickle
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.common import CLIArguments as cli
from ray.train._checkpoint import Checkpoint
from ray.train._internal.session import _TrainingResult
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
def default_policy_agent_mapping(unused_agent_id) -> str:
    return DEFAULT_POLICY_ID