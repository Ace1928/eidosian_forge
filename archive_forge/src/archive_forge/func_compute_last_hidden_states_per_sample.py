import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
def compute_last_hidden_states_per_sample(self, hidden_states: Tuple[Tuple[torch.Tensor]], beam_indices: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
        Computes the last hidden states.

        Parameters:
            hidden_states (`Tuple[Tuple[torch.Tensor]]`):
                The generated hidden states. Tuple (one element for each generated token) of tuples (one element for
                each layer of the decoder) of torch.FloatTensor of shape (batch_size*num_beams*num_return_sequences,
                generated_length, hidden_size).
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.

        Return:
            `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length, hidden_size)`
            containing
                the last hidden states.
        ```"""
    last_hidden_states = torch.concat([hidden_states[-1] for hidden_states in hidden_states], dim=1)
    if beam_indices is None:
        return last_hidden_states
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices.clone()[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]
    beam_indices[beam_indices_mask] = 0
    beam_indices = beam_indices.unsqueeze(-1)
    beam_indices = beam_indices.expand(-1, -1, last_hidden_states.shape[-1])
    last_hidden_states = torch.gather(last_hidden_states, 0, beam_indices)
    return last_hidden_states