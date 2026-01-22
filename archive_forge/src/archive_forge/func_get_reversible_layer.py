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
@staticmethod
def get_reversible_layer(config) -> Tuple[nn.Module, nn.Module]:
    ln_factory = _get_ln_factory(config.dim_model, config.residual_norm_style, residual=False, use_triton=config.use_triton, normalization=config.normalization)
    mha = build_multi_head_attention(config.multi_head_config)
    feedforward = build_feedforward(asdict(config.feedforward_config))
    reversible_f = ln_factory(mha)
    reversible_g = ln_factory(feedforward)
    return (reversible_f, reversible_g)