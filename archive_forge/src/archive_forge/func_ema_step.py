import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def ema_step(self, inputs, length, past_state=None):
    if length == 1:
        return self.one_ema_step(inputs, past_state=past_state)
    damping_factor, previous_timestep_weight = self.get_ema_coefficients()
    vander = torch.arange(length + 1).to(damping_factor).view(1, 1, length + 1) * torch.log(previous_timestep_weight)
    vander = torch.exp(vander)
    if past_state is not None:
        past_ema_proj = vander[:, :, 1:] * (self.kernel_projection_matrix * self.scale).unsqueeze(-1)
        past_ema_state = torch.einsum('bdn,dnl->bdl', past_state, past_ema_proj)
        past_vandermonde = vander[:, :, -1] * past_state
    else:
        past_ema_state = None
        past_vandermonde = None
    vander = vander[:, :, :-1]
    kernel = damping_factor * self.ema_expansion_matrix * vander
    kernel_proj = torch.einsum('dnl,dn->dl', kernel, self.kernel_projection_matrix * self.scale)
    ema_output = self.fft_convolution(inputs, kernel_proj, length=length)[..., 0:length]
    ema_output = ema_output.type_as(inputs)
    if past_ema_state is not None:
        ema_output = ema_output + past_ema_state
    updated_hidden_state = torch.einsum('bdl,dnl->bdn', inputs, torch.flip(kernel, dims=[2]))
    if past_vandermonde is not None:
        updated_hidden_state = updated_hidden_state + past_vandermonde
    return (ema_output.permute(2, 0, 1), updated_hidden_state)