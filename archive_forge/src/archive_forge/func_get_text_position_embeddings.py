import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
def get_text_position_embeddings(self, text_features: Tensor, text_position_embedding: Optional[torch.Tensor], text_position_ids: Optional[torch.Tensor]) -> Tensor:
    batch_size, seq_length, _ = text_features.shape
    if text_position_embedding is None and text_position_ids is None:
        text_position_embedding = torch.arange(seq_length, device=text_features.device)
        text_position_embedding = text_position_embedding.float()
        text_position_embedding = text_position_embedding.unsqueeze(0).unsqueeze(-1)
        text_position_embedding = text_position_embedding.repeat(batch_size, 1, 1)
        text_position_embedding = get_sine_pos_embed(text_position_embedding, num_pos_feats=self.d_model, exchange_xy=False)
    if text_position_ids is not None:
        text_position_embedding = get_sine_pos_embed(text_position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False)
    return text_position_embedding