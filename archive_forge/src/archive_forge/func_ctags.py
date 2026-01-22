from __future__ import annotations
import os
import subprocess
from pathlib import Path
import typing as T
def ctags() -> int:
    ls = ls_as_bytestream()
    return subprocess.run(['ctags', '-L-'], input=ls).returncode