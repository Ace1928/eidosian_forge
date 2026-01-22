from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override
from pytorch_lightning.utilities.model_summary.model_summary import (
def deepspeed_param_size(p: torch.nn.Parameter) -> int:
    assert hasattr(p, 'ds_numel')
    return p.ds_numel