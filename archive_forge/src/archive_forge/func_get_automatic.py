import os
import sys
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def get_automatic(class_type: Union[Type, Tuple[Type, ...]], register: Dict[str, Tuple[Union[Type, Tuple[Type, ...]], str]]) -> List[str]:
    automatic = []
    for key, (base_class, link_to) in register.items():
        if not isinstance(base_class, tuple):
            base_class = (base_class,)
        if link_to == 'AUTOMATIC' and any((issubclass(c, class_type) for c in base_class)):
            automatic.append(key)
    return automatic