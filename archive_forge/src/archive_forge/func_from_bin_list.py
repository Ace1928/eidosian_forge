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
@classmethod
def from_bin_list(cls, env: 'Environment', for_machine: MachineChoice, name: str) -> 'ExternalProgram':
    command = env.lookup_binary_entry(for_machine, name)
    if command is None:
        return NonExistingExternalProgram()
    return cls.from_entry(name, command)