import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
def get_qa_logit_layer(self) -> nn.Module:
    """
        Returns the linear layer that produces question answering logits

        Returns:
            `nn.Module`: A torch module mapping the question answering prediction hidden states. `None`: A NoneType
            object if Lxmert does not have the visual answering head.
        """
    if hasattr(self, 'answer_head'):
        return self.answer_head.logit_fc[-1]