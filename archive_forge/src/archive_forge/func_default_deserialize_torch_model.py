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
def default_deserialize_torch_model(model: Any, state_bytes: bytes, device: 'torch.device') -> Any:
    """Deserializes the parameters of the wrapped PyTorch model and
    moves it to the specified device.

    model:
        Wrapped PyTorch model.
    state_bytes:
        Serialized parameters as a byte stream.
    device:
        PyTorch device to which the model is bound.

    Returns:
        The deserialized model.
    """
    filelike = BytesIO(state_bytes)
    filelike.seek(0)
    model.load_state_dict(torch.load(filelike, map_location=device))
    model.to(device)
    return model