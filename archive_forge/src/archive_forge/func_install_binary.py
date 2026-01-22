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
def install_binary(cls, binary: str, flags: List[str]=None):
    if cls.get_binary_path(binary):
        return
    args = PkgInstall.get_args(binary, flags)
    return subprocess.check_call(args, stdout=subprocess.DEVNULL)