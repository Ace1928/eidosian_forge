import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig
def get_conditional_embeddings(self, batch_size: int=None, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, conditional_pixel_values: Optional[torch.Tensor]=None):
    if input_ids is not None:
        if len(input_ids) != batch_size:
            raise ValueError('Make sure to pass as many prompt texts as there are query images')
        with torch.no_grad():
            conditional_embeddings = self.clip.get_text_features(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    elif conditional_pixel_values is not None:
        if len(conditional_pixel_values) != batch_size:
            raise ValueError('Make sure to pass as many prompt images as there are query images')
        with torch.no_grad():
            conditional_embeddings = self.clip.get_image_features(conditional_pixel_values)
    else:
        raise ValueError('Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`')
    return conditional_embeddings