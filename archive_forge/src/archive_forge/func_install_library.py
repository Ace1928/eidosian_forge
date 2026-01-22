import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
@classmethod
def install_library(cls, library: str, upgrade: bool=True):
    pip_exec = [sys.executable, '-m', 'pip', 'install']
    if '=' not in library or upgrade:
        pip_exec.append('--upgrade')
    pip_exec.append(library)
    return subprocess.check_call(pip_exec, stdout=subprocess.DEVNULL)