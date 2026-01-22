import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
def get_image_wide_positional_embeddings(self):
    size = self.config.prompt_encoder_config.image_embedding_size
    target_device = self.shared_image_embedding.positional_embedding.device
    target_dtype = self.shared_image_embedding.positional_embedding.dtype
    grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / size
    x_embed = x_embed / size
    positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
    return positional_embedding.permute(2, 0, 1).unsqueeze(0)