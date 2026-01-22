import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
@lru_cache()
def is_torch_bf16_available_on_device(device):
    if not is_torch_available():
        return False
    import torch
    if device == 'cuda':
        return is_torch_bf16_gpu_available()
    try:
        x = torch.zeros(2, 2, dtype=torch.bfloat16).to(device)
        _ = x @ x
    except:
        return False
    return True