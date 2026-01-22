import copy
import dataclasses
import itertools
import os
from typing import Any, Callable, Dict, List
import torch
import torch._lazy as lazy
import torch._lazy.metrics as metrics
from torch import fx
from torch._lazy import computation, debug as lazy_debug
from torch._lazy.tensor_factory_functions import tensor_factory_functions
def hasDeviceArg(args, kwargs):
    return any((isinstance(arg, torch.device) for arg in itertools.chain(args, kwargs.values())))