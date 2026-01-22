from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_sys_root(self) -> T.Optional[str]:
    sys_root = self.properties.get('sys_root', None)
    assert sys_root is None or isinstance(sys_root, str)
    return sys_root