import sys
import re
import os
from collections import namedtuple
from numba.core.config import IS_WIN32
from numba.misc.findlib import find_lib, find_file
def get_nvidia_nvvm_ctk():
    """Return path to directory containing the NVVM shared library.
    """
    is_conda_env = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    if not is_conda_env:
        return
    libdir = os.path.join(sys.prefix, 'nvvm', _cudalib_path())
    if not os.path.exists(libdir) or not os.path.isdir(libdir):
        libdir = os.path.join(sys.prefix, 'Library', 'nvvm', _cudalib_path())
        if not os.path.exists(libdir) or not os.path.isdir(libdir):
            return
    paths = find_lib('nvvm', libdir=libdir)
    if not paths:
        return
    return os.path.dirname(max(paths))