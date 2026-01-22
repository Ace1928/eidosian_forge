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
def forward_tokens(self, music_tokens, music_tokens_conds=[], metadata=None, get_preds=False, get_attn_weights=False):
    """
        Applies a forward pass using the conditioning tokens. Different from the classic forward as it does not use the
        vqvae's encoding layers.
        """
    if get_attn_weights:
        self.prior.transformer.set_record_attn(get_attn_weights)
    audio_conditioning, metadata_conditioning, lyric_tokens = self.get_cond(music_tokens_conds, metadata)
    if self.is_encoder_decoder:
        tokens, audio_conditioning = self.prior_preprocess([lyric_tokens, music_tokens], [None, audio_conditioning])
        (encoder_loss, next_token_prediction_loss), preds = self.prior(tokens, audio_conditioning, metadata_conditioning, get_sep_loss=True, get_preds=get_preds)
    else:
        last_encoder_hidden_states = self.get_encoder_states(lyric_tokens)
        encoder_loss = self.get_encoder_loss(last_encoder_hidden_states, lyric_tokens)
        next_token_prediction_loss, preds = self.prior(music_tokens, audio_conditioning, metadata_conditioning, last_encoder_hidden_states, get_preds=get_preds)
    loss = self.encoder_loss_fraction * encoder_loss * self.nb_relevant_lyric_tokens / self.total_loss_dims
    loss += next_token_prediction_loss * self.next_token_prediction_loss_dims / self.total_loss_dims
    metrics = {'bpd': next_token_prediction_loss.clone().detach(), 'encoder_loss': encoder_loss.clone().detach(), 'next_token_prediction_loss': next_token_prediction_loss.clone().detach()}
    if get_preds:
        metrics['preds'] = preds.clone().detach()
    if get_attn_weights:
        saved_attn_weights = self.prior.transformer.saved_attn_weights
        self.prior.transformer.set_record_attn(False)
        return saved_attn_weights
    else:
        return (loss, metrics)