from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_cmake_defaults(self) -> bool:
    if 'cmake_defaults' not in self.properties:
        return True
    res = self.properties['cmake_defaults']
    assert isinstance(res, bool)
    return res