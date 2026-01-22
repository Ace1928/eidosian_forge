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
def register_empty_buffer(module, name, buffer, persistent=True):
    old_register_buffer(module, name, buffer, persistent=persistent)
    if buffer is not None:
        module._buffers[name] = module._buffers[name].to(device)