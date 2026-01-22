import math
import warnings
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
def dora_init(self, adapter_name: str) -> None:
    lora_A = self.lora_A[adapter_name]
    lora_B = self.lora_B[adapter_name]
    scaling = self.scaling[adapter_name]
    with gather_params_ctx(self.get_base_layer()):
        weight = self.get_base_layer().weight
        lora_weight = lora_B.weight @ lora_A.weight
        weight_norm = self._get_weight_norm(weight, lora_weight, scaling)
    self.lora_magnitude_vector = nn.ParameterDict()
    self.lora_magnitude_vector[adapter_name] = nn.Parameter(weight_norm, requires_grad=True)
    self.adapter_layer_names = self.adapter_layer_names[:] + ('lora_magnitude_vector',)