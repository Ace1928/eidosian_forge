import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any((inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor))):
        warnings.warn('None of the inputs have requires_grad=True. Gradients will be None')