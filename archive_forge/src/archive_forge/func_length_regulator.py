import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
def length_regulator(encoded_embeddings, duration_labels, speaking_speed=1.0):
    """
    Length regulator for feed-forward Transformer.

    This is the length regulator module described in `FastSpeech: Fast, Robust and Controllable Text to Speech`
    https://arxiv.org/pdf/1905.09263.pdf. The length regulator expands char or phoneme-level embedding features to
    frame-level by repeating each feature based on the corresponding predicted durations.

    Args:
        encoded_embeddings (`torch.Tensor` of shape `(batch_size, max_text_length, embedding_dim)`):
            Batch of sequences of char or phoneme embeddings.
        duration_labels (`torch.LongTensor` of shape `(batch_size, time)`):
            Batch of durations of each frame.
        speaking_speed (`float`, *optional*, defaults to 1.0):
            Value to control speed of speech.

    Returns:
        `torch.Tensor`:
            Replicated input tensor based on durations (batch_size, time*, embedding_dim).
    """
    if speaking_speed <= 0:
        raise ValueError('`speaking_speed` must be greater than 0.')
    elif speaking_speed != 1.0:
        duration_labels = torch.round(duration_labels.float() * speaking_speed).long()
    if duration_labels.sum() == 0:
        duration_labels[duration_labels.sum(dim=1).eq(0)] = 1
    max_len = torch.sum(duration_labels, dim=1).max()
    hidden_states = torch.zeros((encoded_embeddings.size(0), max_len, encoded_embeddings.size(2)), dtype=torch.float, device=encoded_embeddings.device)
    for i, (encoded_embedding, target_duration) in enumerate(zip(encoded_embeddings, duration_labels)):
        repeated = torch.repeat_interleave(encoded_embedding, target_duration, dim=0)
        hidden_states[i, :repeated.size(0)] = repeated
    return hidden_states