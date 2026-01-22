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
def is_torch_mlu_available(check_device=False):
    """Checks if `torch_mlu` is installed and potentially if a MLU is in the environment"""
    if not _torch_available or importlib.util.find_spec('torch_mlu') is None:
        return False
    import torch
    import torch_mlu
    from ..dependency_versions_table import deps
    deps['deepspeed'] = 'deepspeed-mlu>=0.10.1'
    if check_device:
        try:
            _ = torch.mlu.device_count()
            return torch.mlu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, 'mlu') and torch.mlu.is_available()