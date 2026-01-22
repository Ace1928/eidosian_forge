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
@staticmethod
def distogram(coords, min_bin, max_bin, num_bins):
    boundaries = torch.linspace(min_bin, max_bin, num_bins - 1, device=coords.device)
    boundaries = boundaries ** 2
    N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
    b = CA - N
    c = C - CA
    a = b.cross(c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
    bins = torch.sum(dists > boundaries, dim=-1)
    return bins