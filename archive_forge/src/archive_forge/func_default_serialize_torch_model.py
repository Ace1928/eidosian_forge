import contextlib
import itertools
from io import BytesIO
from typing import Any, Callable, Dict, Optional, cast
import srsly
from ..backends import CupyOps, context_pools, get_current_ops, set_gpu_allocator
from ..compat import torch
from ..optimizers import Optimizer
from ..types import ArgsKwargs, FloatsXd
from ..util import (
from .pytorch_grad_scaler import PyTorchGradScaler
from .shim import Shim
def default_serialize_torch_model(model: Any) -> bytes:
    """Serializes the parameters of the wrapped PyTorch model to bytes.

    model:
        Wrapped PyTorch model.

    Returns:
        A `bytes` object that encapsulates the serialized model parameters.
    """
    filelike = BytesIO()
    torch.save(model.state_dict(), filelike)
    filelike.seek(0)
    return filelike.getvalue()