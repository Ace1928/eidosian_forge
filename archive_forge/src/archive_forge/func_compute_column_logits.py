import enum
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import (
from .configuration_tapas import TapasConfig
def compute_column_logits(sequence_output, column_output_weights, column_output_bias, cell_index, cell_mask, allow_empty_column_selection):
    """
    Computes the column logits.

    Args:
        sequence_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        column_output_weights (`torch.FloatTensor` of shape `(hidden_size)`):
            Weights of the linear layer for column selection.
        column_output_bias (`torch.FloatTensor` of shape `()`):
            Bias of the linear layer for column selection.
        cell_index (`ProductIndexMap`):
            Index that groups tokens into cells.
        cell_mask (`torch.FloatTensor` of shape `(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).
        allow_empty_column_selection (`bool`):
            Whether to allow not to select any column

    Returns:
        column_logits (`torch.FloatTensor`of shape `(batch_size, max_num_cols)`): Tensor containing the column logits
        for every example in the batch.
    """
    token_logits = torch.einsum('bsj,j->bs', sequence_output, column_output_weights) + column_output_bias
    cell_logits, cell_logits_index = reduce_mean(token_logits, cell_index)
    column_index = cell_index.project_inner(cell_logits_index)
    column_logits, out_index = reduce_sum(cell_logits * cell_mask, column_index)
    cell_count, _ = reduce_sum(cell_mask, column_index)
    column_logits /= cell_count + EPSILON_ZERO_DIVISION
    is_padding = torch.logical_and(cell_count < 0.5, ~torch.eq(out_index.indices, 0))
    column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(is_padding, dtype=torch.float32, device=is_padding.device)
    if not allow_empty_column_selection:
        column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(torch.eq(out_index.indices, 0), dtype=torch.float32, device=out_index.indices.device)
    return column_logits