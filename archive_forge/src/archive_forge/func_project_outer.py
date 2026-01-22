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
def project_outer(self, index):
    """Projects an index with the same index set onto the outer components."""
    indices = torch.div(index.indices, self.inner_index.num_segments, rounding_mode='floor').type(torch.long)
    return IndexMap(indices=indices, num_segments=self.outer_index.num_segments, batch_dims=index.batch_dims)