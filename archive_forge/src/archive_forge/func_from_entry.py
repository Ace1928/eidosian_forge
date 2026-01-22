from __future__ import annotations
import functools
import os
import shutil
import stat
import sys
import re
import typing as T
from pathlib import Path
from . import mesonlib
from . import mlog
from .mesonlib import MachineChoice, OrderedSet
@staticmethod
def from_entry(name: str, command: T.Union[str, T.List[str]]) -> 'ExternalProgram':
    if isinstance(command, list):
        if len(command) == 1:
            command = command[0]
    if isinstance(command, list) or os.path.isabs(command):
        if isinstance(command, str):
            command = [command]
        return ExternalProgram(name, command=command, silent=True)
    assert isinstance(command, str)
    return ExternalProgram(command, silent=True)