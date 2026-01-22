import logging
from typing import Any, Dict
import torch
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model, wavlm_model
def transform_wavlm_encoder_state(state: Dict[str, Any], encoder_num_layers: int):
    """Converts WavLM encoder state from HuggingFace format. In particular, concatenates linear projection weights and
    biases to align with the structure of ``torch.nn.MultiheadAttention``.
    """
    for i in range(encoder_num_layers):
        q_proj_bias = state.pop(f'layers.{i}.attention.q_proj.bias')
        k_proj_bias = state.pop(f'layers.{i}.attention.k_proj.bias')
        v_proj_bias = state.pop(f'layers.{i}.attention.v_proj.bias')
        q_proj_weight = state.pop(f'layers.{i}.attention.q_proj.weight')
        k_proj_weight = state.pop(f'layers.{i}.attention.k_proj.weight')
        v_proj_weight = state.pop(f'layers.{i}.attention.v_proj.weight')
        state[f'layers.{i}.attention.attention.in_proj_bias'] = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias))
        state[f'layers.{i}.attention.attention.in_proj_weight'] = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight))
        state[f'layers.{i}.attention.attention.out_proj.weight'] = state.pop(f'layers.{i}.attention.out_proj.weight')
        state[f'layers.{i}.attention.attention.out_proj.bias'] = state.pop(f'layers.{i}.attention.out_proj.bias')