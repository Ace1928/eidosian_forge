import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def read_submodules(path: str) -> Iterator[Tuple[bytes, bytes, bytes]]:
    """Read a .gitmodules file."""
    cfg = ConfigFile.from_path(path)
    return parse_submodules(cfg)