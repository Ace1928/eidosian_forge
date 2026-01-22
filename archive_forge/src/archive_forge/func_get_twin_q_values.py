import gymnasium as gym
from typing import Optional, List, Dict
from ray.rllib.algorithms.sac.sac_torch_model import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override, force_list
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@override(SACTorchModel)
def get_twin_q_values(self, model_out: TensorType, state_in: List[TensorType], seq_lens: TensorType, actions: Optional[TensorType]=None) -> TensorType:
    return self._get_q_value(model_out, actions, self.twin_q_net, state_in, seq_lens)