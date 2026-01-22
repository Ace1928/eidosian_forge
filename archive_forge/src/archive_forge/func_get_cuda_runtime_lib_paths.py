import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def get_cuda_runtime_lib_paths(candidate_paths: Set[Path]) -> Set[Path]:
    paths = set()
    for libname in CUDA_RUNTIME_LIBS:
        for path in candidate_paths:
            try:
                if (path / libname).is_file():
                    paths.add(path / libname)
            except PermissionError:
                pass
    return paths