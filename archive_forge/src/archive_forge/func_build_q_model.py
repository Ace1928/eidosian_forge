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
def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
    """Builds one of the (twin) Q-nets used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own Q-nets. Alternatively, simply set `custom_model` within the
        top level SAC `q_model_config` config key to make this default implementation
        of `build_q_model` use your custom Q-nets.

        Returns:
            TorchModelV2: The TorchModelV2 Q-net sub-model.
        """
    self.concat_obs_and_actions = False
    if self.discrete:
        input_space = obs_space
    else:
        orig_space = getattr(obs_space, 'original_space', obs_space)
        if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
            input_space = Box(float('-inf'), float('inf'), shape=(orig_space.shape[0] + action_space.shape[0],))
            self.concat_obs_and_actions = True
        else:
            input_space = gym.spaces.Tuple([orig_space, action_space])
    model = ModelCatalog.get_model_v2(input_space, action_space, num_outputs, q_model_config, framework='torch', name=name)
    return model