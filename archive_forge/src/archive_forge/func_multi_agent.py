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
def multi_agent(self, *, policies=NotProvided, algorithm_config_overrides_per_module: Optional[Dict[ModuleID, PartialAlgorithmConfigDict]]=NotProvided, policy_map_capacity: Optional[int]=NotProvided, policy_mapping_fn: Optional[Callable[[AgentID, 'OldEpisode'], PolicyID]]=NotProvided, policies_to_train: Optional[Union[Container[PolicyID], Callable[[PolicyID, SampleBatchType], bool]]]=NotProvided, policy_states_are_swappable: Optional[bool]=NotProvided, observation_fn: Optional[Callable]=NotProvided, count_steps_by: Optional[str]=NotProvided, replay_mode=DEPRECATED_VALUE, policy_map_cache=DEPRECATED_VALUE) -> 'AlgorithmConfig':
    """Sets the config's multi-agent settings.

        Validates the new multi-agent settings and translates everything into
        a unified multi-agent setup format. For example a `policies` list or set
        of IDs is properly converted into a dict mapping these IDs to PolicySpecs.

        Args:
            policies: Map of type MultiAgentPolicyConfigDict from policy ids to either
                4-tuples of (policy_cls, obs_space, act_space, config) or PolicySpecs.
                These tuples or PolicySpecs define the class of the policy, the
                observation- and action spaces of the policies, and any extra config.
            algorithm_config_overrides_per_module: Only used if
                `_enable_new_api_stack=True`.
                A mapping from ModuleIDs to per-module AlgorithmConfig override dicts,
                which apply certain settings,
                e.g. the learning rate, from the main AlgorithmConfig only to this
                particular module (within a MultiAgentRLModule).
                You can create override dicts by using the `AlgorithmConfig.overrides`
                utility. For example, to override your learning rate and (PPO) lambda
                setting just for a single RLModule with your MultiAgentRLModule, do:
                config.multi_agent(algorithm_config_overrides_per_module={
                "module_1": PPOConfig.overrides(lr=0.0002, lambda_=0.75),
                })
            policy_map_capacity: Keep this many policies in the "policy_map" (before
                writing least-recently used ones to disk/S3).
            policy_mapping_fn: Function mapping agent ids to policy ids. The signature
                is: `(agent_id, episode, worker, **kwargs) -> PolicyID`.
            policies_to_train: Determines those policies that should be updated.
                Options are:
                - None, for training all policies.
                - An iterable of PolicyIDs that should be trained.
                - A callable, taking a PolicyID and a SampleBatch or MultiAgentBatch
                and returning a bool (indicating whether the given policy is trainable
                or not, given the particular batch). This allows you to have a policy
                trained only on certain data (e.g. when playing against a certain
                opponent).
            policy_states_are_swappable: Whether all Policy objects in this map can be
                "swapped out" via a simple `state = A.get_state(); B.set_state(state)`,
                where `A` and `B` are policy instances in this map. You should set
                this to True for significantly speeding up the PolicyMap's cache lookup
                times, iff your policies all share the same neural network
                architecture and optimizer types. If True, the PolicyMap will not
                have to garbage collect old, least recently used policies, but instead
                keep them in memory and simply override their state with the state of
                the most recently accessed one.
                For example, in a league-based training setup, you might have 100s of
                the same policies in your map (playing against each other in various
                combinations), but all of them share the same state structure
                (are "swappable").
            observation_fn: Optional function that can be used to enhance the local
                agent observations to include more state. See
                rllib/evaluation/observation_function.py for more info.
            count_steps_by: Which metric to use as the "batch size" when building a
                MultiAgentBatch. The two supported values are:
                "env_steps": Count each time the env is "stepped" (no matter how many
                multi-agent actions are passed/how many multi-agent observations
                have been returned in the previous step).
                "agent_steps": Count each individual agent step as one step.

        Returns:
            This updated AlgorithmConfig object.
        """
    if policies is not NotProvided:
        for pid in policies:
            validate_policy_id(pid, error=True)
        if isinstance(policies, dict):
            for pid, spec in policies.items():
                if not isinstance(spec, PolicySpec):
                    if not isinstance(spec, (list, tuple)) or len(spec) != 4:
                        raise ValueError(f'Policy specs must be tuples/lists of (cls or None, obs_space, action_space, config), got {spec} for PolicyID={pid}')
                elif not isinstance(spec.config, (AlgorithmConfig, dict)) and spec.config is not None:
                    raise ValueError(f'Multi-agent policy config for {pid} must be a dict or AlgorithmConfig object, but got {type(spec.config)}!')
        self.policies = policies
    if algorithm_config_overrides_per_module is not NotProvided:
        self.algorithm_config_overrides_per_module = algorithm_config_overrides_per_module
    if policy_map_capacity is not NotProvided:
        self.policy_map_capacity = policy_map_capacity
    if policy_mapping_fn is not NotProvided:
        if isinstance(policy_mapping_fn, dict):
            policy_mapping_fn = from_config(policy_mapping_fn)
        self.policy_mapping_fn = policy_mapping_fn
    if observation_fn is not NotProvided:
        self.observation_fn = observation_fn
    if policy_map_cache != DEPRECATED_VALUE:
        deprecation_warning(old='AlgorithmConfig.multi_agent(policy_map_cache=..)', error=True)
    if replay_mode != DEPRECATED_VALUE:
        deprecation_warning(old='AlgorithmConfig.multi_agent(replay_mode=..)', new="AlgorithmConfig.training(replay_buffer_config={'replay_mode': ..})", error=True)
    if count_steps_by is not NotProvided:
        if count_steps_by not in ['env_steps', 'agent_steps']:
            raise ValueError(f'config.multi_agent(count_steps_by=..) must be one of [env_steps|agent_steps], not {count_steps_by}!')
        self.count_steps_by = count_steps_by
    if policies_to_train is not NotProvided:
        assert isinstance(policies_to_train, (list, set, tuple)) or callable(policies_to_train) or policies_to_train is None, 'ERROR: `policies_to_train` must be a [list|set|tuple] or a callable taking PolicyID and SampleBatch and returning True|False (trainable or not?) or None (for always training all policies).'
        if isinstance(policies_to_train, (list, set, tuple)):
            if len(policies_to_train) == 0:
                logger.warning('`config.multi_agent(policies_to_train=..)` is empty! Make sure - if you would like to learn at least one policy - to add its ID to that list.')
        self.policies_to_train = policies_to_train
    if policy_states_are_swappable is not NotProvided:
        self.policy_states_are_swappable = policy_states_are_swappable
    return self