import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.rank_zero import WarningCache
def parse_batch_shape(batch: Any) -> Union[str, List]:
    if hasattr(batch, 'shape'):
        return list(batch.shape)
    if isinstance(batch, (list, tuple)):
        return [parse_batch_shape(el) for el in batch]
    return UNKNOWN_SIZE