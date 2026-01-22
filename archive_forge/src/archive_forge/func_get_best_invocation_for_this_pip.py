import itertools
import os
import shutil
import sys
from typing import List, Optional
from pip._internal.cli.main import main
from pip._internal.utils.compat import WINDOWS
def get_best_invocation_for_this_pip() -> str:
    """Try to figure out the best way to invoke pip in the current environment."""
    binary_directory = 'Scripts' if WINDOWS else 'bin'
    binary_prefix = os.path.join(sys.prefix, binary_directory)
    path_parts = os.path.normcase(os.environ.get('PATH', '')).split(os.pathsep)
    exe_are_in_PATH = os.path.normcase(binary_prefix) in path_parts
    if exe_are_in_PATH:
        for exe_name in _EXECUTABLE_NAMES:
            found_executable = shutil.which(exe_name)
            binary_executable = os.path.join(binary_prefix, exe_name)
            if found_executable and os.path.exists(binary_executable) and os.path.samefile(found_executable, binary_executable):
                return exe_name
    return f'{get_best_invocation_for_this_python()} -m pip'