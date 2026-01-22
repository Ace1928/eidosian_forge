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
def project_inner(self, index):
    """Projects an index with the same index set onto the inner components."""
    return IndexMap(indices=torch.fmod(index.indices, self.inner_index.num_segments).type(torch.float).floor().type(torch.long), num_segments=self.inner_index.num_segments, batch_dims=index.batch_dims)