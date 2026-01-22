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
def prior_preprocess(self, tokens, conds):
    """
        Shifts the input tokens to account for the dictionary merge. The embed_dim_shift give by how much the music
        tokens should be shifted by. It is equal to `lyric_vocab_size`.
        """
    batch_size = tokens[0].shape[0]
    for i in range(len(tokens)):
        tokens[i] = (tokens[i] + int(self.embed_dim_shift[i])).view(batch_size, -1)
    for i in range(len(conds)):
        if conds[i] is None:
            conds[i] = torch.zeros((batch_size, self.input_shapes[i], self.width), dtype=tokens[0].dtype, device=tokens[0].device)
    return (torch.cat(tokens, dim=1), torch.cat(conds, dim=1))