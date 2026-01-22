import inspect
import logging
from queue import Queue
from functools import wraps
from typing import Callable, Dict, List
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
def depends_on(a: Callable, b: Callable):
    if a == that and b == this:
        return False
    return True