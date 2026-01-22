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
@ExperimentalAPI
def rl_module(self, *, rl_module_spec: Optional[ModuleSpec]=NotProvided, _enable_rl_module_api: Optional[bool]=NotProvided) -> 'AlgorithmConfig':
    """Sets the config's RLModule settings.

        Args:
            rl_module_spec: The RLModule spec to use for this config. It can be either
                a SingleAgentRLModuleSpec or a MultiAgentRLModuleSpec. If the
                observation_space, action_space, catalog_class, or the model config is
                not specified it will be inferred from the env and other parts of the
                algorithm config object.

        Returns:
            This updated AlgorithmConfig object.
        """
    if rl_module_spec is not NotProvided:
        self._rl_module_spec = rl_module_spec
    if _enable_rl_module_api is not NotProvided:
        deprecation_warning(old='AlgorithmConfig.rl_module(_enable_rl_module_api=True|False)', new='AlgorithmConfig.experimental(_enable_new_api_stack=True|False)', error=True)
    return self