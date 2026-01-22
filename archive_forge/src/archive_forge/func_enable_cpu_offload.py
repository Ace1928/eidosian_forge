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
def enable_cpu_offload(self, gpu_id: Optional[int]=0):
    """
        Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This
        method moves one whole sub-model at a time to the GPU when it is used, and the sub-model remains in GPU until
        the next sub-model runs.

        Args:
            gpu_id (`int`, *optional*, defaults to 0):
                GPU id on which the sub-models will be loaded and offloaded.
        """
    if is_accelerate_available():
        from accelerate import cpu_offload_with_hook
    else:
        raise ImportError('`enable_model_cpu_offload` requires `accelerate`.')
    device = torch.device(f'cuda:{gpu_id}')
    if self.device.type != 'cpu':
        self.to('cpu')
        torch.cuda.empty_cache()
    self.semantic.input_embeds_layer, _ = cpu_offload_with_hook(self.semantic.input_embeds_layer, device)
    hook = None
    for cpu_offloaded_model in [self.semantic, self.coarse_acoustics, self.fine_acoustics]:
        _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)
    self.fine_acoustics_hook = hook
    _, hook = cpu_offload_with_hook(self.codec_model, device, prev_module_hook=hook)
    self.codec_model_hook = hook