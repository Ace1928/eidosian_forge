import gymnasium as gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.policy import eager_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import (
def before_loss_init_wrapper(policy, obs_space, action_space, config):
    if before_loss_init:
        before_loss_init(policy, obs_space, action_space, config)
    if extra_action_out_fn is None or policy._is_tower:
        extra_action_fetches = {}
    else:
        extra_action_fetches = extra_action_out_fn(policy)
    if hasattr(policy, '_extra_action_fetches'):
        policy._extra_action_fetches.update(extra_action_fetches)
    else:
        policy._extra_action_fetches = extra_action_fetches