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
def experimental(self, *, _enable_new_api_stack: Optional[bool]=NotProvided, _tf_policy_handles_more_than_one_loss: Optional[bool]=NotProvided, _disable_preprocessor_api: Optional[bool]=NotProvided, _disable_action_flattening: Optional[bool]=NotProvided, _disable_execution_plan_api: Optional[bool]=NotProvided, _disable_initialize_loss_from_dummy_batch: Optional[bool]=NotProvided) -> 'AlgorithmConfig':
    """Sets the config's experimental settings.

        Args:
            _enable_new_api_stack: Enables the new API stack, which will use RLModule
                (instead of ModelV2) as well as the multi-GPU capable Learner API
                (instead of using Policy to compute loss and update the model).
            _tf_policy_handles_more_than_one_loss: Experimental flag.
                If True, TFPolicy will handle more than one loss/optimizer.
                Set this to True, if you would like to return more than
                one loss term from your `loss_fn` and an equal number of optimizers
                from your `optimizer_fn`. In the future, the default for this will be
                True.
            _disable_preprocessor_api: Experimental flag.
                If True, no (observation) preprocessor will be created and
                observations will arrive in model as they are returned by the env.
                In the future, the default for this will be True.
            _disable_action_flattening: Experimental flag.
                If True, RLlib will no longer flatten the policy-computed actions into
                a single tensor (for storage in SampleCollectors/output files/etc..),
                but leave (possibly nested) actions as-is. Disabling flattening affects:
                - SampleCollectors: Have to store possibly nested action structs.
                - Models that have the previous action(s) as part of their input.
                - Algorithms reading from offline files (incl. action information).
            _disable_execution_plan_api: Experimental flag.
                If True, the execution plan API will not be used. Instead,
                a Algorithm's `training_iteration` method will be called as-is each
                training iteration.

        Returns:
            This updated AlgorithmConfig object.
        """
    if _enable_new_api_stack is not NotProvided:
        self._enable_new_api_stack = _enable_new_api_stack
        if _enable_new_api_stack is True and self.exploration_config:
            self.__prior_exploration_config = self.exploration_config
            self.exploration_config = {}
        elif _enable_new_api_stack is False and (not self.exploration_config):
            if self.__prior_exploration_config is not None:
                self.exploration_config = self.__prior_exploration_config
                self.__prior_exploration_config = None
            else:
                logger.warning('config._enable_new_api_stack was set to False, but no prior exploration config was found to be restored.')
    if _tf_policy_handles_more_than_one_loss is not NotProvided:
        self._tf_policy_handles_more_than_one_loss = _tf_policy_handles_more_than_one_loss
    if _disable_preprocessor_api is not NotProvided:
        self._disable_preprocessor_api = _disable_preprocessor_api
    if _disable_action_flattening is not NotProvided:
        self._disable_action_flattening = _disable_action_flattening
    if _disable_execution_plan_api is not NotProvided:
        self._disable_execution_plan_api = _disable_execution_plan_api
    if _disable_initialize_loss_from_dummy_batch is not NotProvided:
        self._disable_initialize_loss_from_dummy_batch = _disable_initialize_loss_from_dummy_batch
    return self