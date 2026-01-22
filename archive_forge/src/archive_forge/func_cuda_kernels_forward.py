import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import DynamicCache  # we need __iter__ and __len__ of pkv
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import (
from .configuration_jamba import JambaConfig
def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: HybridMambaAttentionDynamicCache=None):
    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = cache_params is not None and cache_params.has_previous_state and (seq_len == 1) and (cache_params.conv_states[self.layer_idx].shape[0] == cache_params.ssm_states[self.layer_idx].shape[0] == batch_size)
    projected_states = self.in_proj(hidden_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)
    conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    if use_precomputed_states:
        hidden_states = causal_conv1d_update(hidden_states.squeeze(-1), cache_params.conv_states[self.layer_idx], conv_weights, self.conv1d.bias, self.activation)
        hidden_states = hidden_states.unsqueeze(-1)
    else:
        if cache_params is not None:
            conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
            cache_params.conv_states[self.layer_idx].copy_(conv_states)
        hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1)
    time_step = self.dt_layernorm(time_step)
    B = self.b_layernorm(B)
    C = self.c_layernorm(C)
    time_proj_bias = self.dt_proj.bias
    self.dt_proj.bias = None
    discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
    self.dt_proj.bias = time_proj_bias
    A = -torch.exp(self.A_log.float())
    time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
    if use_precomputed_states:
        scan_outputs = selective_state_update(cache_params.ssm_states[self.layer_idx], hidden_states[..., 0], discrete_time_step[..., 0], A, B[:, 0], C[:, 0], self.D, gate[..., 0], time_proj_bias, dt_softplus=True).unsqueeze(-1)
    else:
        scan_outputs, ssm_state = selective_scan_fn(hidden_states, discrete_time_step, A, B.transpose(1, 2), C.transpose(1, 2), self.D.float(), gate, time_proj_bias, delta_softplus=True, return_last_state=True)
        if ssm_state is not None and cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
    contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
    return contextualized_states