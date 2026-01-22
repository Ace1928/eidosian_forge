import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
def get_action_model_outputs(self, model_out: TensorType, state_in: List[TensorType]=None, seq_lens: TensorType=None) -> (TensorType, List[TensorType]):
    """Returns distribution inputs and states given the output of
        policy.model().

        For continuous action spaces, these will be the mean/stddev
        distribution inputs for the (SquashedGaussian) action distribution.
        For discrete action spaces, these will be the logits for a categorical
        distribution.

        Args:
            model_out: Feature outputs from the model layers
                (result of doing `model(obs)`).
            state_in List(TensorType): State input for recurrent cells
            seq_lens: Sequence lengths of input- and state
                sequences

        Returns:
            TensorType: Distribution inputs for sampling actions.
        """

    def concat_obs_if_necessary(obs: TensorStructType):
        """Concat model outs if they come as original tuple observations."""
        if isinstance(obs, (list, tuple)):
            obs = torch.cat(obs, dim=-1)
        elif isinstance(obs, dict):
            obs = torch.cat([torch.unsqueeze(val, 1) if len(val.shape) == 1 else val for val in tree.flatten(obs.values())], dim=-1)
        return obs
    if state_in is None:
        state_in = []
    if isinstance(model_out, dict) and 'obs' in model_out:
        if isinstance(self.action_model.obs_space, Box):
            model_out['obs'] = concat_obs_if_necessary(model_out['obs'])
        return self.action_model(model_out, state_in, seq_lens)
    else:
        if isinstance(self.action_model.obs_space, Box):
            model_out = concat_obs_if_necessary(model_out)
        return self.action_model({'obs': model_out}, state_in, seq_lens)