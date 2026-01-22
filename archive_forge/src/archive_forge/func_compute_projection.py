import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
def compute_projection(pair, mask, a=True, chunked=True):
    need_transpose = self._outgoing ^ a
    if not chunked:
        p = compute_projection_helper(pair, mask, a)
        if need_transpose:
            p = p.transpose(-1, -2)
    else:
        linear_g = self.linear_a_g if a else self.linear_b_g
        c = linear_g.bias.shape[-1]
        out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
        p = pair.new_zeros(out_shape)
        for i in range(0, pair.shape[-3], inplace_chunk_size):
            pair_chunk = pair[..., i:i + inplace_chunk_size, :, :]
            pair_chunk = compute_projection_helper(pair[..., i:i + inplace_chunk_size, :, :], mask[..., i:i + inplace_chunk_size, :, :], a)
            if need_transpose:
                pair_chunk = pair_chunk.transpose(-1, -2)
                p[..., i:i + inplace_chunk_size] = pair_chunk
            else:
                p[..., i:i + inplace_chunk_size, :] = pair_chunk
            del pair_chunk
    return p