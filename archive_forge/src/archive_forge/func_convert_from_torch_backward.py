from typing import Any, Callable, Dict, Optional, Tuple, cast
from ..compat import torch
from ..config import registry
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim
from ..types import ArgsKwargs, Floats3d, Padded
from ..util import (
def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Padded:
    dX = torch2xp(d_inputs.args[0])
    return Padded(dX, size_at_t, lengths, indices)