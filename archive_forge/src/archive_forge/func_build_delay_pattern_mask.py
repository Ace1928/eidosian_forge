import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
def build_delay_pattern_mask(self, input_ids: torch.LongTensor, pad_token_id: int, max_length: int=None):
    """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:
        - [P, -1, -1, -1, -1, P, P, P]
        - [P, P, -1, -1, -1, -1, P, P]
        - [P, P, P, -1, -1, -1, -1, P]
        - [P, P, P, P, -1, -1, -1, -1]
        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:
        - [P, a, b, -1, -1, P, P, P]
        - [P, P, c, d, -1, -1, P, P]
        - [P, P, P, e, f, -1, -1, P]
        - [P, P, P, P, g, h, -1, -1]
        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
    input_ids = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
    bsz, num_codebooks, seq_len = input_ids.shape
    max_length = max_length if max_length is not None else self.generation_config.max_length
    input_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device) * -1
    channel_codebooks = num_codebooks // 2 if self.config.audio_channels == 2 else num_codebooks
    if max_length < 2 * channel_codebooks - 1:
        return (input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1))
    for codebook in range(channel_codebooks):
        if self.config.audio_channels == 1:
            input_ids_shifted[:, codebook, codebook:seq_len + codebook] = input_ids[:, codebook]
        else:
            input_ids_shifted[:, 2 * codebook, codebook:seq_len + codebook] = input_ids[:, 2 * codebook]
            input_ids_shifted[:, 2 * codebook + 1, codebook:seq_len + codebook] = input_ids[:, 2 * codebook + 1]
    delay_pattern = torch.triu(torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal=max_length - channel_codebooks + 1)
    delay_pattern = delay_pattern + torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))
    if self.config.audio_channels == 2:
        delay_pattern = delay_pattern.repeat_interleave(2, dim=0)
    mask = ~delay_pattern.to(input_ids.device)
    input_ids = mask * input_ids_shifted + ~mask * pad_token_id
    first_codebook_ids = input_ids[:, 0, :]
    start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
    if len(start_ids) > 0:
        first_start_id = min(start_ids)
    else:
        first_start_id = seq_len
    pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
    input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
    return (input_ids, pattern_mask)