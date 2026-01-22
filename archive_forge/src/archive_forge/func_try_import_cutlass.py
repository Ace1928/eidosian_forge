import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
import sympy
import torch
from ...codecache import cache_dir
from ...config import cuda as inductor_cuda_config
from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version
@functools.lru_cache(None)
def try_import_cutlass() -> bool:
    cutlass_py_full_path = os.path.abspath(os.path.join(inductor_cuda_config.cutlass_dir, 'python/cutlass_library'))
    tmp_cutlass_py_full_path = os.path.abspath(os.path.join(cache_dir(), 'torch_cutlass_library'))
    dst_link = os.path.join(tmp_cutlass_py_full_path, 'cutlass_library')
    if os.path.isdir(cutlass_py_full_path):
        if tmp_cutlass_py_full_path not in sys.path:
            if os.path.exists(dst_link):
                assert os.path.islink(dst_link), f'{dst_link} is not a symlink. Try to remove {dst_link} manually and try again.'
                assert os.path.realpath(os.readlink(dst_link)) == os.path.realpath(cutlass_py_full_path), f'Symlink at {dst_link} does not point to {cutlass_py_full_path}'
            else:
                os.makedirs(tmp_cutlass_py_full_path, exist_ok=True)
                os.symlink(cutlass_py_full_path, dst_link)
            sys.path.append(tmp_cutlass_py_full_path)
        try:
            import cutlass_library.generator
            import cutlass_library.library
            import cutlass_library.manifest
            return True
        except ImportError as e:
            log.debug('Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.', str(e))
    else:
        log.debug('Failed to import CUTLASS packages: CUTLASS repo does not exist: %s', cutlass_py_full_path)
    return False