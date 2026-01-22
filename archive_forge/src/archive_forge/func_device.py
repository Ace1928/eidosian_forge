import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
@property
def device(self) -> torch.device:
    """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
    if not hasattr(self.semantic, '_hf_hook'):
        return get_parameter_device(self)
    for module in self.semantic.modules():
        if hasattr(module, '_hf_hook') and hasattr(module._hf_hook, 'execution_device') and (module._hf_hook.execution_device is not None):
            return torch.device(module._hf_hook.execution_device)