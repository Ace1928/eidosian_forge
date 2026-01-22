from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
def update_model_kwargs_for_generation(outputs, model_kwargs):
    if 'past_key_values' in outputs:
        model_kwargs['past_key_values'] = outputs.past_key_values
    else:
        model_kwargs['past_key_values'] = None
    if 'token_type_ids' in model_kwargs:
        token_type_ids = model_kwargs['token_type_ids']
        model_kwargs['token_type_ids'] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)
    if 'attention_mask' in model_kwargs:
        attention_mask = model_kwargs['attention_mask']
        model_kwargs['attention_mask'] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    if 'image_attention_mask' in model_kwargs:
        image_attention_mask = model_kwargs['image_attention_mask']
        last_mask = image_attention_mask[:, -1, :].unsqueeze(1)
        model_kwargs['image_attention_mask'] = last_mask
    model_kwargs['image_hidden_states'] = outputs.image_hidden_states
    return model_kwargs