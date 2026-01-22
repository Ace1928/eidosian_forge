from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
def check_learning_achieved(tune_results: 'tune.ResultGrid', min_value, evaluation=False, metric: str='episode_reward_mean'):
    """Throws an error if `min_reward` is not reached within tune_results.

    Checks the last iteration found in tune_results for its
    "episode_reward_mean" value and compares it to `min_reward`.

    Args:
        tune_results: The tune.Tuner().fit() returned results object.
        min_reward: The min reward that must be reached.

    Raises:
        ValueError: If `min_reward` not reached.
    """
    recorded_values = [row[metric] if not evaluation else row[f'evaluation/{metric}'] for _, row in tune_results.get_dataframe().iterrows()]
    best_value = max(recorded_values)
    if best_value < min_value:
        raise ValueError(f'`{metric}` of {min_value} not reached!')
    print(f'`{metric}` of {min_value} reached! ok')