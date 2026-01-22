import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import UdopConfig
from transformers.modeling_outputs import (
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..deprecated._archive_maps import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
def get_relative_position(self, positions):
    context_position = positions[:, :, None]
    memory_position = positions[:, None, :]
    relative_position = memory_position - context_position
    if self.augmentation and self.training:
        relative_position *= random.uniform(*AUGMENTATION_RANGE)
    relative_position *= self.scaling_factor
    return relative_position.to(torch.long)