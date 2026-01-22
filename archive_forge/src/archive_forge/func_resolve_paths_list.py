import ctypes as ct
import errno
import os
from pathlib import Path
import platform
from typing import Set, Union
from warnings import warn
import torch
from .env_vars import get_potentially_lib_path_containing_env_vars
def resolve_paths_list(paths_list_candidate: str) -> Set[Path]:
    """
    Searches a given environmental var for the CUDA runtime library,
    i.e. `libcudart.so`.
    """
    return remove_non_existent_dirs(extract_candidate_paths(paths_list_candidate))