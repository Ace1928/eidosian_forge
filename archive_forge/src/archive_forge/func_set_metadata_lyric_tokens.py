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
def set_metadata_lyric_tokens(self, labels):
    """
        Processes the full labels to only retreive the relevant lyric tokens and keep the metadata conditioning tokens.
        """
    if self.nb_relevant_lyric_tokens > 0:
        tokens_list = torch.zeros((labels.shape[0], self.nb_relevant_lyric_tokens), dtype=torch.long, device=labels.device)
        indices_list = []
        for idx in range(labels.shape[0]):
            full_tokens = labels.clone()[:, 4 + self.metadata_embedding.max_nb_genres:]
            total_length, offset, duration = (labels[idx, 0], labels[idx, 1], labels[idx, 2])
            tokens, indices = get_relevant_lyric_tokens(full_tokens, self.nb_relevant_lyric_tokens, total_length, offset, duration)
            tokens_list[idx, :] = tokens
            indices_list.append(indices)
        return (torch.cat((labels[:, :4 + self.metadata_embedding.max_nb_genres], tokens_list), dim=-1), indices_list)
    else:
        return (labels, None)