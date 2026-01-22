import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import logging
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import (
def compute_td_error(obs_t, act_t, rew_t, obs_tp1, terminateds_mask, importance_weights):
    input_dict = self._lazy_tensor_dict({SampleBatch.CUR_OBS: obs_t, SampleBatch.ACTIONS: act_t, SampleBatch.REWARDS: rew_t, SampleBatch.NEXT_OBS: obs_tp1, SampleBatch.TERMINATEDS: terminateds_mask, PRIO_WEIGHTS: importance_weights})
    actor_critic_loss(self, self.model, None, input_dict)
    return self.model.tower_stats['td_error']