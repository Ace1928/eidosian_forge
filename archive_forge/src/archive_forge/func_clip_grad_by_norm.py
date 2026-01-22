from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, Optional
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from typing_extensions import get_args, override
import pytorch_lightning as pl
from lightning_fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning_fabric.plugins.precision.fsdp import _PRECISION_INPUT
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
from pytorch_lightning.plugins.precision.precision import Precision
from pytorch_lightning.utilities.exceptions import MisconfigurationException
@override
def clip_grad_by_norm(self, *_: Any, **__: Any) -> None:
    raise MisconfigurationException(f"`gradient_clip_algorithm='norm'` is currently not supported for `{self.__class__.__name__}`")