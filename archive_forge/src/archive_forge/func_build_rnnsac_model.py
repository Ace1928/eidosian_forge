import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple, Type, Union
import copy
import ray
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.sac import SACTorchPolicy
from ray.rllib.algorithms.sac.rnnsac_torch_model import RNNSACTorchModel
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import huber_loss, sequence_mask
from ray.rllib.utils.typing import ModelInputDict, TensorType, AlgorithmConfigDict
def build_rnnsac_model(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> ModelV2:
    """Constructs the necessary ModelV2 for the Policy and returns it.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SAC's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    num_outputs = int(np.product(obs_space.shape))
    policy_model_config = copy.deepcopy(config['model'])
    policy_model_config.update(config['policy_model_config'])
    q_model_config = copy.deepcopy(config['model'])
    q_model_config.update(config['q_model_config'])
    default_model_cls = RNNSACTorchModel
    model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(model, default_model_cls)
    policy.target_model = ModelCatalog.get_model_v2(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=config['model'], framework=config['framework'], default_model=default_model_cls, name='target_sac_model', policy_model_config=policy_model_config, q_model_config=q_model_config, twin_q=config['twin_q'], initial_alpha=config['initial_alpha'], target_entropy=config['target_entropy'])
    assert isinstance(policy.target_model, default_model_cls)
    return model