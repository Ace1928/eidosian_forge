import os  # noqa: C101
import sys
from typing import Any, Dict, TYPE_CHECKING
import torch
from torch.utils._config_module import install_config_module
def decide_compile_threads():
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform or it's a fbcode build
    3. decide by the number of CPU cores
    """
    if 'TORCHINDUCTOR_COMPILE_THREADS' in os.environ:
        return int(os.environ['TORCHINDUCTOR_COMPILE_THREADS'])
    elif sys.platform == 'win32' or is_fbcode():
        return 1
    else:
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        assert cpu_count
        return min(32, cpu_count)