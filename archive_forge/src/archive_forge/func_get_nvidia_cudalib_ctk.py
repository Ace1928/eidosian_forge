import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_nvidia_cudalib_ctk():
    """Return path to directory containing the shared libraries of cudatoolkit.
    """
    nvvm_ctk = get_nvidia_nvvm_ctk()
    if not nvvm_ctk:
        return
    env_dir = os.path.dirname(os.path.dirname(nvvm_ctk))
    subdir = 'bin' if IS_WIN32 else 'lib'
    return os.path.join(env_dir, subdir)