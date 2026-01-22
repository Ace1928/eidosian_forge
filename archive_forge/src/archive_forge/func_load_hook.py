import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig
from ..deprecated._archive_maps import MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
def load_hook(self, state_dict, prefix, *args):
    for k in state_dict:
        if 'embedding.' in k:
            state_dict[k.replace('embedding.', 'embeddings.')] = state_dict.pop(k)
            break