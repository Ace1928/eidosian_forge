import abc
import logging
from typing import Tuple, Union
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.specs.checker import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.util import log_once
@override(Model)
def get_num_parameters(self) -> Tuple[int, int]:
    num_all_params = sum((int(np.prod(p.size())) for p in self.parameters()))
    trainable_params = filter(lambda p: p.requires_grad, self.parameters())
    num_trainable_params = sum((int(np.prod(p.size())) for p in trainable_params))
    return (num_trainable_params, num_all_params - num_trainable_params)