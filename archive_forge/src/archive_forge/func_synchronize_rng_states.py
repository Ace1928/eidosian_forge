import random
from typing import List, Optional, Union
import numpy as np
import torch
from ..state import AcceleratorState
from .constants import CUDA_DISTRIBUTED_TYPES
from .dataclasses import DistributedType, RNGType
from .imports import is_npu_available, is_torch_xla_available, is_xpu_available
def synchronize_rng_states(rng_types: List[Union[str, RNGType]], generator: Optional[torch.Generator]=None):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)