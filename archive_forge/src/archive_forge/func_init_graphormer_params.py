import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
def init_graphormer_params(self, module: Union[nn.Linear, nn.Embedding, GraphormerMultiheadAttention]):
    """
        Initialize the weights specific to the Graphormer Model.
        """
    if isinstance(module, nn.Linear):
        self.normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        self.normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, GraphormerMultiheadAttention):
        self.normal_(module.q_proj.weight.data)
        self.normal_(module.k_proj.weight.data)
        self.normal_(module.v_proj.weight.data)