import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
def fix_speech_decoder_output(self, speech_ids: torch.LongTensor) -> torch.LongTensor:
    """
        This method modifies the output of the decoder model, such as replacing the `eos_token_id` and changing the
        last few tokens of each sequence.

        Args:
            speech_ids (`torch.LongTensor`):
                This refers to the output of the decoder model.
        """
    decoder_fixing_codes = self.config.decoder_config.decoder_fixing_codes
    speech_ids = speech_ids[:, 1:]
    stop_token_indices = torch.where(speech_ids == self.speech_decoder_model.config.eos_token_id, 1, 0)
    speech_ids = torch.masked_fill(speech_ids, mask=stop_token_indices.bool(), value=decoder_fixing_codes[0])
    for i, each_seq_stop_token_index in enumerate(stop_token_indices):
        if each_seq_stop_token_index.sum() == 0:
            continue
        stm = each_seq_stop_token_index.argmax()
        speech_ids[i, stm:] = decoder_fixing_codes[0]
        if stm - 3 < speech_ids.shape[1]:
            speech_ids[i, -3:] = torch.tensor([decoder_fixing_codes[1:]], device=speech_ids.device, dtype=torch.long)
    return speech_ids