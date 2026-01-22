import math
from typing import Optional
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, logging
from ...bert.modeling_bert import BertModel
from .configuration_retribert import RetriBertConfig
def partial_encode(*inputs):
    encoder_outputs = sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = sent_encoder.pooler(sequence_output)
    return pooled_output