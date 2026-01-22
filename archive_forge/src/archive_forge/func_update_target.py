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
def update_target(self, polyak_coef=None):
    polyak_coef = polyak_coef or self.polyak_coef
    model_state_dict = self.q_model.state_dict()
    target_state_dict = self.target_q_model.state_dict()
    model_state_dict = {k: polyak_coef * model_state_dict[k] + (1 - polyak_coef) * v for k, v in target_state_dict.items()}
    self.target_q_model.load_state_dict(model_state_dict)