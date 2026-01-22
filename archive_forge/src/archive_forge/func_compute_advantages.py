from typing import List
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.models.base import STATE_IN
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorType
def compute_advantages(episode: SingleAgentEpisode, last_r: float, gamma: float=0.9, lambda_: float=1.0, use_critic: bool=True, use_gae: bool=True, rewards: TensorType=None, vf_preds: TensorType=None):
    assert SampleBatch.VF_PREDS in episode.extra_model_outputs or not use_critic, 'use_critic=True but values not found'
    assert use_critic or not use_gae, "Can't use gae without using a value function."
    last_r = convert_to_numpy(last_r)
    if rewards is None:
        rewards = episode.rewards
    if vf_preds is None:
        vf_preds = episode.extra_model_outs[SampleBatch.VF_PREDS]
    if use_gae:
        vpred_t = np.concatenate([vf_preds, np.array([last_r])])
        delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
        episode.extra_model_outputs[Postprocessing.ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
        episode.extra_model_outputs[Postprocessing.VALUE_TARGETS] = (episode.extra_model_outputs[Postprocessing.ADVANTAGES] + vf_preds).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate([rewards, np.array([last_r])])
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(np.float32)
        if use_critic:
            episode.extra_model_outputs[Postprocessing.ADVANTAGES] = discounted_returns - vf_preds
            episode.extra_model_outputs[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            episode.extra_model_outputs[Postprocessing.ADVANTAGES] = discounted_returns
            episode.extra_model_outputs[Postprocessing.VALUE_TARGETS] = np.zeros_like(episode.extra_model_outputs[Postprocessing.ADVANTAGES])
    episode.extra_model_outputs[Postprocessing.ADVANTAGES] = episode.extra_model_outputs[Postprocessing.ADVANTAGES].astype(np.float32)
    return episode