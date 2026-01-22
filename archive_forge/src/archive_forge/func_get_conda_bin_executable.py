import logging
import os
import shutil
import subprocess
import hashlib
import json
from typing import Optional, List, Union, Tuple
def get_conda_bin_executable(executable_name: str) -> str:
    """
    Return path to the specified executable, assumed to be discoverable within
    a conda installation.

    The conda home directory (expected to contain a 'bin' subdirectory on
    linux) is configurable via the ``RAY_CONDA_HOME`` environment variable. If
    ``RAY_CONDA_HOME`` is unspecified, try the ``CONDA_EXE`` environment
    variable set by activating conda. If neither is specified, this method
    returns `executable_name`.
    """
    conda_home = os.environ.get(RAY_CONDA_HOME)
    if conda_home:
        if _WIN32:
            candidate = os.path.join(conda_home, '%s.exe' % executable_name)
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.join(conda_home, '%s.bat' % executable_name)
            if os.path.exists(candidate):
                return candidate
        else:
            return os.path.join(conda_home, 'bin/%s' % executable_name)
    else:
        conda_home = '.'
    if 'CONDA_EXE' in os.environ:
        conda_bin_dir = os.path.dirname(os.environ['CONDA_EXE'])
        if _WIN32:
            candidate = os.path.join(conda_home, '%s.exe' % executable_name)
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.join(conda_home, '%s.bat' % executable_name)
            if os.path.exists(candidate):
                return candidate
        else:
            return os.path.join(conda_bin_dir, executable_name)
    if _WIN32:
        return executable_name + '.bat'
    return executable_name