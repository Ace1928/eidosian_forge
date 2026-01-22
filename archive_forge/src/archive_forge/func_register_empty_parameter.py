import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from .hooks import (
from .utils import (
from .utils.other import recursive_getattr
def register_empty_parameter(module, name, param):
    old_register_parameter(module, name, param)
    if param is not None:
        param_cls = type(module._parameters[name])
        kwargs = module._parameters[name].__dict__
        kwargs['requires_grad'] = param.requires_grad
        module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)