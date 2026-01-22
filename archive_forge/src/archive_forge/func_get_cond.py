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
def get_cond(self, music_tokens_conds, metadata):
    """
        Converts the input tokens to input_embeddings. Splits the lyrics form the rest of the metadata. Lyric tokens
        can be None.
        """
    if metadata is not None:
        n_labels = metadata.shape[1] - self.nb_relevant_lyric_tokens
        metadata, lyric_tokens = (metadata[:, :n_labels], metadata[:, n_labels:])
    else:
        metadata, lyric_tokens = (None, None)
    metadata_conditioning, metadata_pos = self.metadata_embedding(metadata) if self.metadata_conditioning else (None, None)
    audio_conditioning = self.embed_tokens(music_tokens_conds) if self.audio_conditioning else metadata_pos
    return (audio_conditioning, metadata_conditioning, lyric_tokens)