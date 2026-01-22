import importlib
import importlib.metadata
import os
import warnings
from functools import lru_cache
import torch
from packaging import version
from packaging.version import parse
from .environment import parse_flag_from_env, str_to_bool
from .versions import compare_versions, is_torch_version
def is_4bit_bnb_available():
    package_exists = _is_package_available('bitsandbytes')
    if package_exists:
        bnb_version = version.parse(importlib.metadata.version('bitsandbytes'))
        return compare_versions(bnb_version, '>=', '0.39.0')
    return False