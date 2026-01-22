from typing import Dict, Any
from ray.rllib.models.utils import get_initializer
from ray.rllib.policy import Policy
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import is_overridden
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gymnasium.spaces import Discrete
def estimate_v(self, batch: SampleBatch) -> TensorType:
    obs = torch.tensor(batch[SampleBatch.OBS], device=self.device)
    with torch.no_grad():
        q_values, _ = self.q_model({'obs': obs}, [], None)
    action_probs = self._compute_action_probs(obs)
    v_values = torch.sum(q_values * action_probs, axis=-1)
    return v_values