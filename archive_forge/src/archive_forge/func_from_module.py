import abc
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Mapping, Any, TYPE_CHECKING, Optional, Type, Dict, Union
import gymnasium as gym
import tree
import ray
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.policy.policy import get_gym_space_from_struct_of_tensors
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.models.specs.checker import (
from ray.rllib.models.distributions import Distribution
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.serialization import (
@classmethod
def from_module(cls, module: 'RLModule') -> 'SingleAgentRLModuleSpec':
    from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
    if isinstance(module, MultiAgentRLModule):
        raise ValueError('MultiAgentRLModule cannot be converted to SingleAgentRLModuleSpec.')
    return SingleAgentRLModuleSpec(module_class=type(module), observation_space=module.config.observation_space, action_space=module.config.action_space, model_config_dict=module.config.model_config_dict, catalog_class=module.config.catalog_class)