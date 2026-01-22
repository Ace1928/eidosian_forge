import contextlib
import functools
import logging
from unittest.mock import patch
import torch
from torch._dynamo import disable
from torch._dynamo.utils import counters, defake
from torch._functorch.aot_autograd import aot_module_simplified
from torch.utils._python_dispatch import _disable_current_modes
def device_from_inputs(example_inputs) -> torch.device:
    for x in example_inputs:
        if hasattr(x, 'device'):
            return x.device