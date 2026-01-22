import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
def concatenate_mask(self, mask_token, sequence, ids_restore):
    batch_size, seq_length, dim = sequence.shape
    mask_tokens = mask_token.repeat(batch_size, ids_restore.shape[1] - seq_length, 1)
    padded_sequence = torch.cat([sequence, mask_tokens], dim=1)
    padded_sequence = torch.gather(padded_sequence, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim))
    return padded_sequence