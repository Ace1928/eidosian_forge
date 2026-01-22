import copy
import logging
import math
import os
import sys
from typing import (
from packaging import version
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.learner.learner_group_config import LearnerGroupConfig, ModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.learner.learner import TorchCompileWhatToCompile
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import deep_update, merge_dicts
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided, from_config
from ray.rllib.utils.gym import (
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import (
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import (
from ray.tune.logger import Logger
from ray.tune.registry import get_trainable_cls
from ray.tune.result import TRIAL_INFO
from ray.tune.tune import _Config
def update_from_dict(self, config_dict: PartialAlgorithmConfigDict) -> 'AlgorithmConfig':
    """Modifies this AlgorithmConfig via the provided python config dict.

        Warns if `config_dict` contains deprecated keys.
        Silently sets even properties of `self` that do NOT exist. This way, this method
        may be used to configure custom Policies which do not have their own specific
        AlgorithmConfig classes, e.g.
        `ray.rllib.examples.policy.random_policy::RandomPolicy`.

        Args:
            config_dict: The old-style python config dict (PartialAlgorithmConfigDict)
                to use for overriding some properties defined in there.

        Returns:
            This updated AlgorithmConfig object.
        """
    eval_call = {}
    if '_enable_new_api_stack' in config_dict:
        self.experimental(_enable_new_api_stack=config_dict['_enable_new_api_stack'])
    for key, value in config_dict.items():
        key = self._translate_special_keys(key, warn_deprecated=False)
        if key == TRIAL_INFO:
            continue
        if key == '_enable_new_api_stack':
            continue
        elif key == 'multiagent':
            kwargs = {k: value[k] for k in ['policies', 'policy_map_capacity', 'policy_mapping_fn', 'policies_to_train', 'policy_states_are_swappable', 'observation_fn', 'count_steps_by'] if k in value}
            self.multi_agent(**kwargs)
        elif key == 'callbacks_class' and value != NOT_SERIALIZABLE:
            if isinstance(value, str):
                value = deserialize_type(value, error=True)
            self.callbacks(callbacks_class=value)
        elif key == 'env_config':
            self.environment(env_config=value)
        elif key.startswith('evaluation_'):
            eval_call[key] = value
        elif key == 'exploration_config':
            if config_dict.get('_enable_new_api_stack', False):
                self.exploration_config = value
                continue
            if isinstance(value, dict) and 'type' in value:
                value['type'] = deserialize_type(value['type'])
            self.exploration(exploration_config=value)
        elif key == 'model':
            if isinstance(value, dict) and value.get('custom_model'):
                value['custom_model'] = deserialize_type(value['custom_model'])
            self.training(**{key: value})
        elif key == 'optimizer':
            self.training(**{key: value})
        elif key == 'replay_buffer_config':
            if isinstance(value, dict) and 'type' in value:
                value['type'] = deserialize_type(value['type'])
            self.training(**{key: value})
        elif key == 'sample_collector':
            value = deserialize_type(value)
            self.rollouts(sample_collector=value)
        else:
            setattr(self, key, value)
    self.evaluation(**eval_call)
    return self