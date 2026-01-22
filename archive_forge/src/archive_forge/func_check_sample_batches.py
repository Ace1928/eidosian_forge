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
def check_sample_batches(_batch1, _batch2, _policy_id=None):
    unroll_id_1 = _batch1.get('unroll_id', None)
    unroll_id_2 = _batch2.get('unroll_id', None)
    if unroll_id_1 is not None and unroll_id_2 is not None:
        assert unroll_id_1 == unroll_id_2
    batch1_keys = set()
    for k, v in _batch1.items():
        if k == 'unroll_id':
            continue
        check(v, _batch2[k])
        batch1_keys.add(k)
    batch2_keys = set(_batch2.keys())
    batch2_keys.discard('unroll_id')
    _difference = batch1_keys.symmetric_difference(batch2_keys)
    if _policy_id:
        assert not _difference, "SampleBatches for policy with ID {} don't share information on the following information: \n{}".format(_policy_id, _difference)
    else:
        assert not _difference, "SampleBatches don't share information on the following information: \n{}".format(_difference)