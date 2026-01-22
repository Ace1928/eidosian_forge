from typing import TYPE_CHECKING, Any, ContextManager, Dict, Literal, Optional, cast
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import get_args, override
from lightning_fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
@property
def mixed_precision_config(self) -> 'TorchMixedPrecision':
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision
    if self.precision == '16-mixed':
        param_dtype = None if not _TORCH_GREATER_EQUAL_2_0 else torch.float32
        reduce_dtype = buffer_dtype = torch.float16
    elif self.precision == 'bf16-mixed':
        param_dtype = None if not _TORCH_GREATER_EQUAL_2_0 else torch.float32
        reduce_dtype = buffer_dtype = torch.bfloat16
    elif self.precision == '16-true':
        param_dtype = reduce_dtype = buffer_dtype = torch.float16
    elif self.precision == 'bf16-true':
        param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
    elif self.precision == '32-true':
        param_dtype = None if not _TORCH_GREATER_EQUAL_2_0 else torch.float32
        reduce_dtype = buffer_dtype = torch.float32
    else:
        raise ValueError(f'Was unable to infer precision type, received {self.precision!r}.')
    return TorchMixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)