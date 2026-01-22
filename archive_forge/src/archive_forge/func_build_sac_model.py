import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import (
def build_sac_model(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> ModelV2:
    """Constructs the necessary ModelV2 for the Policy and returns it.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SACConfig object.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    policy_model_config = copy.deepcopy(config['model'])
    policy_model_config.update(config['policy_model_config'])
    q_model_config = copy.deepcopy(config['model'])
    q_model_config.update(config['q_model_config'])
    default_model_cls = SACTorchModel if config['framework'] == 'torch' else SACTFModel
    model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=None, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(model, default_model_cls)
    policy.target_model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=None, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='target_sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(policy.target_model, default_model_cls)
    return model