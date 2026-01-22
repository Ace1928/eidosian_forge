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
def patch_tensor_constructor(fn):

    def wrapper(*args, **kwargs):
        kwargs['device'] = device
        return fn(*args, **kwargs)
    return wrapper