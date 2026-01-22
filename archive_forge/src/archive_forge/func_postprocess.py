import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def postprocess(self, latent_states, dequantised_states, x_shape):
    batch_size, time = x_shape
    dequantised_states = dequantised_states.view(batch_size, time, -1).permute(0, 2, 1).contiguous()
    latent_states = latent_states.view(batch_size, time)
    return (latent_states, dequantised_states)