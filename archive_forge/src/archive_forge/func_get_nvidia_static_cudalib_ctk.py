import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_nvidia_static_cudalib_ctk():
    """Return path to directory containing the static libraries of cudatoolkit.
    """
    nvvm_ctk = get_nvidia_nvvm_ctk()
    if not nvvm_ctk:
        return
    if IS_WIN32 and 'Library' not in nvvm_ctk:
        dirs = ('Lib', 'x64')
    else:
        dirs = ('lib',)
    env_dir = os.path.dirname(os.path.dirname(nvvm_ctk))
    return os.path.join(env_dir, *dirs)