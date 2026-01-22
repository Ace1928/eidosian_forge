import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
def span_candidates(masks):
    """
            Generate span candidates.

            Args:
                masks: <bool> [num_retrievals, max_sequence_len]

            Returns:
                starts: <int32> [num_spans] ends: <int32> [num_spans] span_masks: <int32> [num_retrievals, num_spans]
                whether spans locate in evidence block.
            """
    _, max_sequence_len = masks.shape

    def _spans_given_width(width):
        current_starts = torch.arange(max_sequence_len - width + 1, device=masks.device)
        current_ends = torch.arange(width - 1, max_sequence_len, device=masks.device)
        return (current_starts, current_ends)
    starts, ends = zip(*(_spans_given_width(w + 1) for w in range(self.config.max_span_width)))
    starts = torch.cat(starts, 0)
    ends = torch.cat(ends, 0)
    start_masks = torch.index_select(masks, dim=-1, index=starts)
    end_masks = torch.index_select(masks, dim=-1, index=ends)
    span_masks = start_masks * end_masks
    return (starts, ends, span_masks)