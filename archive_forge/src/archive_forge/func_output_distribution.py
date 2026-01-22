import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_autoformer import AutoformerConfig
@torch.jit.ignore
def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
    sliced_params = params
    if trailing_n is not None:
        sliced_params = [p[:, -trailing_n:] for p in params]
    return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)