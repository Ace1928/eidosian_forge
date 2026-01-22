import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator
import torch
from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import NONPYTORCH_DOC_URL
from bitsandbytes.cuda_specs import CUDASpecs
from bitsandbytes.diagnostics.utils import print_dedented
def find_cudart_libraries() -> Iterator[Path]:
    """
    Searches for a cuda installations, in the following order of priority:
        1. active conda env
        2. LD_LIBRARY_PATH
        3. any other env vars, while ignoring those that
            - are known to be unrelated
            - don't contain the path separator `/`

    If multiple libraries are found in part 3, we optimistically try one,
    while giving a warning message.
    """
    candidate_env_vars = get_potentially_lib_path_containing_env_vars()
    for envvar in CUDART_PATH_PREFERRED_ENVVARS:
        if envvar in candidate_env_vars:
            directory = candidate_env_vars[envvar]
            yield from find_cuda_libraries_in_path_list(directory)
            candidate_env_vars.pop(envvar)
    for env_var, value in candidate_env_vars.items():
        yield from find_cuda_libraries_in_path_list(value)