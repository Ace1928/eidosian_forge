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
def get_music_tokens_conds(self, music_tokens, start, end):
    """
        Extracts current level's conditioning music tokens.
        """
    if self.level != 0:
        music_tokens_cond = music_tokens[self.level - 1]
        music_tokens = music_tokens_cond[:, start // self.cond_downsample:end // self.cond_downsample]
        missing_cond_len = self.n_ctx // self.cond_downsample - music_tokens_cond[-1].shape[-1]
        if missing_cond_len > 0:
            init_cond = torch.zeros(1, missing_cond_len).to(music_tokens_cond.device)
            music_tokens_cond = torch.cat((music_tokens_cond, init_cond), dim=-1).long()
        music_tokens_conds = [music_tokens_cond]
    else:
        music_tokens_conds = None
    return music_tokens_conds