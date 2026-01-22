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
def get_encoder_loss(self, last_encoder_hidden_states, target_lyrics):
    """
        Computes the loss for the lyric encoder: next lyric token prediction.
        """
    if self.lyric_conditioning:
        last_encoder_hidden_states = self.encoder.lm_head(last_encoder_hidden_states)
        encoder_loss = nn.functional.cross_entropy(last_encoder_hidden_states.view(-1, self.encoder_dim), target_lyrics.view(-1)) / np.log(2.0)
    else:
        encoder_loss = torch.tensor(0.0, device=last_encoder_hidden_states.device)
    return encoder_loss