import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@ExperimentalAPI
@OverrideToImplementCustomLogic
def make_rl_module(self) -> 'RLModule':
    """Returns the RL Module (only for when RLModule API is enabled.)

        If RLModule API is enabled
        (self.config.experimental(_enable_new_api_stack=True), this method should be
        implemented and should return the RLModule instance to use for this Policy.
        Otherwise, RLlib will error out.
        """
    from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
    if self.__policy_id is None:
        raise ValueError('When using RLModule API, `policy_id` within the policies must be set. This should have happened automatically. If you see this bug, please file a github issue.')
    spec = self.config['__marl_module_spec']
    if isinstance(spec, SingleAgentRLModuleSpec):
        module = spec.build()
    else:
        marl_spec = type(spec)(marl_module_class=spec.marl_module_class, module_specs={self.__policy_id: spec.module_specs[self.__policy_id]})
        marl_module = marl_spec.build()
        module = marl_module[self.__policy_id]
    return module