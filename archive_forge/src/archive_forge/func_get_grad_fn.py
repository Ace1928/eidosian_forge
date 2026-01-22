import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
def get_grad_fn(t):
    if t.requires_grad and t.grad_fn is None:
        return t.expand_as(t).grad_fn.next_functions[0][0]
    else:
        return t.grad_fn