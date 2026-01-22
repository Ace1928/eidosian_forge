from gymnasium.spaces import Discrete, Box, MultiDiscrete, Space
import numpy as np
import tree  # pip install dm_tree
from typing import Union, Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils import force_tuple
from ray.rllib.utils.framework import try_import_tf, try_import_torch, TensorType
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.tf_utils import zero_logps_from_actions
def get_torch_exploration_action(self, action_dist: ActionDistribution, explore: bool):
    if explore:
        req = force_tuple(action_dist.required_model_output_shape(self.action_space, getattr(self.model, 'model_config', None)))
        if len(action_dist.inputs.shape) == len(req) + 1:
            batch_size = action_dist.inputs.shape[0]
            a = np.stack([self.action_space.sample() for _ in range(batch_size)])
        else:
            a = self.action_space.sample()
        action = torch.from_numpy(a).to(self.device)
    else:
        action = action_dist.deterministic_sample()
    logp = torch.zeros((action.size()[0],), dtype=torch.float32, device=self.device)
    return (action, logp)