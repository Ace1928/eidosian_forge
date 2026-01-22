from typing import Any, Callable, Dict, Optional, Tuple, cast
from ..compat import torch
from ..config import registry
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim
from ..types import ArgsKwargs, Floats3d, Padded
from ..util import (
def convert_for_torch_backward(dYp: Padded) -> ArgsKwargs:
    dYtorch = xp2torch(dYp.data, requires_grad=True, device=shim.device)
    return ArgsKwargs(args=(Ytorch,), kwargs={'grad_tensors': dYtorch})