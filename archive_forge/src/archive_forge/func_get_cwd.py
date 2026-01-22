import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
@staticmethod
def get_cwd(*paths, string: bool=True) -> Union[str, pathlib.Path]:
    if not paths:
        return pathlib.Path.cwd().as_posix() if string else pathlib.Path.cwd()
    if string:
        return pathlib.Path.cwd().joinpath(*paths).as_posix()
    return pathlib.Path.cwd().joinpath(*paths)