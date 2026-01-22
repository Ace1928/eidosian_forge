from typing import Any, Dict, List, Optional
import torch
from torch.nn.parameter import Parameter
from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
def get_scaled_act_names(self) -> List[str]:
    return []