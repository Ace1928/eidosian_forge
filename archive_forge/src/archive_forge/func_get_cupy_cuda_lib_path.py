import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def get_cupy_cuda_lib_path():
    """Returns the directory where CUDA external libraries are installed.

    This environment variable only affects wheel installations.

    Shared libraries are looked up from
    `$CUPY_CUDA_LIB_PATH/$CUDA_VER/$LIB_NAME/$LIB_VER/{lib,lib64,bin}`,
    e.g., `~/.cupy/cuda_lib/11.2/cudnn/8.1.1/lib64/libcudnn.so.8.1.1`.

    The default $CUPY_CUDA_LIB_PATH is `~/.cupy/cuda_lib`.
    """
    cupy_cuda_lib_path = os.environ.get('CUPY_CUDA_LIB_PATH', None)
    if cupy_cuda_lib_path is None:
        return os.path.expanduser('~/.cupy/cuda_lib')
    return os.path.abspath(cupy_cuda_lib_path)