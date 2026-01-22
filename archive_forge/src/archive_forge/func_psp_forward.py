import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_beit import BeitConfig
def psp_forward(self, inputs):
    x = inputs[-1]
    psp_outs = [x]
    psp_outs.extend(self.psp_modules(x))
    psp_outs = torch.cat(psp_outs, dim=1)
    output = self.bottleneck(psp_outs)
    return output