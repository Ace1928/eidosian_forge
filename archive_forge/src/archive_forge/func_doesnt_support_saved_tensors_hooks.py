import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def doesnt_support_saved_tensors_hooks(f):
    message = "torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case."

    @functools.wraps(f)
    def fn(*args, **kwargs):
        with torch.autograd.graph.disable_saved_tensors_hooks(message):
            return f(*args, **kwargs)
    return fn