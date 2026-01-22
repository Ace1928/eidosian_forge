import copy
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_marian import MarianConfig
def resize_decoder_token_embeddings(self, new_num_tokens):
    if self.config.share_encoder_decoder_embeddings:
        raise ValueError('`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` is `True`. Please use `resize_token_embeddings` instead.')
    old_embeddings = self.model.get_decoder_input_embeddings()
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    self.model.set_decoder_input_embeddings(new_embeddings)
    if self.get_output_embeddings() is not None and (not self.config.tie_word_embeddings):
        old_lm_head = self.get_output_embeddings()
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        self.set_output_embeddings(new_lm_head)
    model_embeds = self.model.get_decoder_input_embeddings()
    if new_num_tokens is None:
        return model_embeds
    self.config.decoder_vocab_size = new_num_tokens
    self.tie_weights()
    self._resize_final_logits_bias(new_num_tokens)
    return model_embeds