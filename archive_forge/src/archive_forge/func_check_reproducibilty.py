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
def check_reproducibilty(algo_class: Type['Algorithm'], algo_config: 'AlgorithmConfig', *, fw_kwargs: Dict[str, Any], training_iteration: int=1) -> None:
    """Check if the algorithm is reproducible across different testing conditions:

        frameworks: all input frameworks
        num_gpus: int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        num_workers: 0 (only local workers) or
                     4 ((1) local workers + (4) remote workers)
        num_envs_per_worker: 2

    Args:
        algo_class: Algorithm class to test.
        algo_config: Base config to use for the algorithm.
        fw_kwargs: Framework iterator keyword arguments.
        training_iteration: Number of training iterations to run.

    Returns:
        None

    Raises:
        It raises an AssertionError if the algorithm is not reproducible.
    """
    from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
    from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
    stop_dict = {'training_iteration': training_iteration}
    for num_workers in [0, 2]:
        algo_config = algo_config.debugging(seed=42).resources(num_gpus=int(os.environ.get('RLLIB_NUM_GPUS', '0')), num_gpus_per_learner_worker=int(os.environ.get('RLLIB_NUM_GPUS', '0'))).rollouts(num_rollout_workers=num_workers, num_envs_per_worker=2)
        for fw in framework_iterator(algo_config, **fw_kwargs):
            print(f'Testing reproducibility of {algo_class.__name__} with {num_workers} workers on fw = {fw}')
            print('/// config')
            pprint.pprint(algo_config.to_dict())
            results1 = tune.Tuner(algo_class, param_space=algo_config.to_dict(), run_config=air.RunConfig(stop=stop_dict, verbose=1)).fit()
            results1 = results1.get_best_result().metrics
            results2 = tune.Tuner(algo_class, param_space=algo_config.to_dict(), run_config=air.RunConfig(stop=stop_dict, verbose=1)).fit()
            results2 = results2.get_best_result().metrics
            check(results1['hist_stats'], results2['hist_stats'])
            if algo_config._enable_new_api_stack:
                check(results1['info'][LEARNER_INFO][DEFAULT_POLICY_ID], results2['info'][LEARNER_INFO][DEFAULT_POLICY_ID])
            else:
                check(results1['info'][LEARNER_INFO][DEFAULT_POLICY_ID]['learner_stats'], results2['info'][LEARNER_INFO][DEFAULT_POLICY_ID]['learner_stats'])