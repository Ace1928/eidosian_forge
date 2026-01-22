import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
def forward_prediction_heads(self, output, mask_features, attention_mask_target_size):
    decoder_output = self.decoder_norm(output)
    decoder_output = decoder_output.transpose(0, 1)
    outputs_class = self.class_embed(decoder_output)
    mask_embed = self.mask_embed(decoder_output)
    outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
    attention_mask = nn.functional.interpolate(outputs_mask, size=attention_mask_target_size, mode='bilinear', align_corners=False)
    attention_mask = (attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
    attention_mask = attention_mask.detach()
    return (outputs_class, outputs_mask, attention_mask)