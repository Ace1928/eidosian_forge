from functools import partial
import numpy as np
import gymnasium as gym
import logging
import tree
from typing import Dict, List, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.typing import (
def policy_actions_repeat(model, action_dist, obs, num_repeat=1):
    batch_size = tf.shape(tree.flatten(obs)[0])[0]
    obs_temp = tree.map_structure(lambda t: _repeat_tensor(t, num_repeat), obs)
    logits, _ = model.get_action_model_outputs(obs_temp)
    policy_dist = action_dist(logits, model)
    actions, logp_ = policy_dist.sample_logp()
    logp = tf.expand_dims(logp_, -1)
    return (actions, tf.reshape(logp, [batch_size, num_repeat, 1]))