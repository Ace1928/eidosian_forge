import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
def quantize(self, hidden_states):
    embed = self.embed.t()
    scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
    dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
    embed_ind = dist.max(dim=-1).indices
    return embed_ind