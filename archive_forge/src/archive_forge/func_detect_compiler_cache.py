from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
@staticmethod
def detect_compiler_cache() -> T.List[str]:
    cache = BinaryTable.detect_sccache()
    if cache:
        return cache
    return BinaryTable.detect_ccache()