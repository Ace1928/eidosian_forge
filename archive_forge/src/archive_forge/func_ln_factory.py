import logging
from dataclasses import asdict
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers._deprecation_warning import deprecated_function
from xformers.components import (
from xformers.components.attention import AttentionMask
from xformers.components.feedforward import build_feedforward
from xformers.components.positional_embedding import build_positional_embedding
from xformers.components.residual import get_deepnorm_coefficients
from xformers.components.simplicial_embedding import SimplicialEmbedding
from xformers.factory.block_configs import (
def ln_factory(sublayer: nn.Module):
    return get_layer_wrapper(d_model, sublayer, residual_norm_style, residual, residual_scale)