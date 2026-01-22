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
@lru_cache
def is_torch_xpu_available(check_device=False):
    """Checks if `intel_extension_for_pytorch` is installed and potentially if a XPU is in the environment"""
    if not is_ipex_available():
        return False
    import intel_extension_for_pytorch
    import torch
    if check_device:
        try:
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, 'xpu') and torch.xpu.is_available()