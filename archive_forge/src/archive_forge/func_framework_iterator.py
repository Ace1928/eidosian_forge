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
def framework_iterator(config: Optional['AlgorithmConfig']=None, frameworks: Sequence[str]=('tf2', 'tf', 'torch'), session: bool=False, time_iterations: Optional[dict]=None) -> Union[str, Tuple[str, Optional['tf1.Session']]]:
    """An generator that allows for looping through n frameworks for testing.

    Provides the correct config entries ("framework") as well
    as the correct eager/non-eager contexts for tf/tf2.

    Args:
        config: An optional config dict or AlgorithmConfig object. This will be modified
            (value for "framework" changed) depending on the iteration.
        frameworks: A list/tuple of the frameworks to be tested.
            Allowed are: "tf2", "tf", "torch", and None.
        session: If True and only in the tf-case: Enter a tf.Session()
            and yield that as second return value (otherwise yield (fw, None)).
            Also sets a seed (42) on the session to make the test
            deterministic.
        time_iterations: If provided, will write to the given dict (by
            framework key) the times in seconds that each (framework's)
            iteration takes.

    Yields:
        If `session` is False: The current framework [tf2|tf|torch] used.
        If `session` is True: A tuple consisting of the current framework
        string and the tf1.Session (if fw="tf", otherwise None).
    """
    config = config or {}
    frameworks = [frameworks] if isinstance(frameworks, str) else list(frameworks)
    for fw in frameworks:
        if fw == 'tf' and config.get('_enable_new_api_stack', False):
            logger.warning('framework_iterator skipping tf (new API stack configured)!')
            continue
        if fw == 'torch' and (not torch):
            logger.warning('framework_iterator skipping torch (not installed)!')
            continue
        if fw != 'torch' and (not tf):
            logger.warning('framework_iterator skipping {} (tf not installed)!'.format(fw))
            continue
        elif fw == 'tf2' and tfv != 2:
            logger.warning('framework_iterator skipping tf2.x (tf version is < 2.0)!')
            continue
        elif fw == 'jax' and (not jax):
            logger.warning('framework_iterator skipping JAX (not installed)!')
            continue
        assert fw in ['tf2', 'tf', 'torch', 'jax', None]
        sess = None
        if fw == 'tf' and session is True:
            sess = tf1.Session()
            sess.__enter__()
            tf1.set_random_seed(42)
        if isinstance(config, dict):
            config['framework'] = fw
        else:
            config.framework(fw)
        eager_ctx = None
        if fw == 'tf2':
            eager_ctx = eager_mode()
            eager_ctx.__enter__()
            assert tf1.executing_eagerly()
        elif fw == 'tf':
            assert not tf1.executing_eagerly()
        print(f'framework={fw}')
        time_started = time.time()
        yield (fw if session is False else (fw, sess))
        if time_iterations is not None:
            time_total = time.time() - time_started
            time_iterations[fw] = time_total
            print(f'.. took {time_total}sec')
        if eager_ctx:
            eager_ctx.__exit__(None, None, None)
        elif sess:
            sess.__exit__(None, None, None)