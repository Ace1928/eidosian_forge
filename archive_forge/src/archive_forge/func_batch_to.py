from abc import ABC
from functools import partial
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from lightning_fabric.utilities.types import _DEVICE
def batch_to(data: Any) -> Any:
    kwargs = {}
    if isinstance(data, Tensor) and isinstance(device, torch.device) and (device.type not in _BLOCKING_DEVICE_TYPES):
        kwargs['non_blocking'] = True
    data_output = data.to(device, **kwargs)
    if data_output is not None:
        return data_output
    return data