import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
@PublicAPI(stability='stable')
def get_device() -> Union[torch.device, List[torch.device]]:
    """Gets the correct torch device configured for this process.

    Returns a list of devices if more than 1 GPU per worker
    is requested.

    Assumes that `CUDA_VISIBLE_DEVICES` is set and is a
    superset of the `ray.get_gpu_ids()`.

    Example:
        >>> # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
        >>> # ray.get_gpu_ids() == [3]
        >>> # torch.cuda.is_available() == True
        >>> # get_device() == torch.device("cuda:0")

        >>> # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
        >>> # ray.get_gpu_ids() == [4]
        >>> # torch.cuda.is_available() == True
        >>> # get_device() == torch.device("cuda:4")

        >>> # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
        >>> # ray.get_gpu_ids() == [4,5]
        >>> # torch.cuda.is_available() == True
        >>> # get_device() == torch.device("cuda:4")
    """
    from ray.air._internal import torch_utils
    record_extra_usage_tag(TagKey.TRAIN_TORCH_GET_DEVICE, '1')
    return torch_utils.get_device()