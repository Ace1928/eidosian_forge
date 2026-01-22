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
def get_delta_weight(self, adapter) -> torch.Tensor:
    """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
    device = self.lora_B[adapter].weight.device
    dtype = self.lora_A[adapter].weight.dtype
    cast_to_fp32 = device.type == 'cpu' and dtype == torch.float16
    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight
    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()
    if self.get_base_layer().weight.size()[2:4] == (1, 1):
        output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
    else:
        output_tensor = F.conv2d(weight_A.permute(1, 0, 2, 3), weight_B).permute(1, 0, 2, 3) * self.scaling[adapter]
    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)
        self.lora_A[adapter].weight.data = weight_A.to(dtype)
        self.lora_B[adapter].weight.data = weight_B.to(dtype)
    return output_tensor