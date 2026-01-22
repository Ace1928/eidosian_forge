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
def is_megatron_lm_available():
    if str_to_bool(os.environ.get('ACCELERATE_USE_MEGATRON_LM', 'False')) == 1:
        package_exists = importlib.util.find_spec('megatron') is not None
        if package_exists:
            try:
                megatron_version = parse(importlib.metadata.version('megatron-lm'))
                return compare_versions(megatron_version, '>=', '2.2.0')
            except Exception as e:
                warnings.warn(f'Parse Megatron version failed. Exception:{e}')
                return False