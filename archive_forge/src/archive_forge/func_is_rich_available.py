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
def is_rich_available():
    if _is_package_available('rich'):
        if 'ACCELERATE_DISABLE_RICH' in os.environ:
            warnings.warn('`ACCELERATE_DISABLE_RICH` is deprecated and will be removed in v0.22.0 and deactivated by default. Please use `ACCELERATE_ENABLE_RICH` if you wish to use `rich`.')
            return not parse_flag_from_env('ACCELERATE_DISABLE_RICH', False)
        return parse_flag_from_env('ACCELERATE_ENABLE_RICH', False)
    return False