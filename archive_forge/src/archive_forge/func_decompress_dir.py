from __future__ import annotations
import os
import shutil
import warnings
from gzip import GzipFile
from pathlib import Path
from typing import TYPE_CHECKING
from monty.io import zopen
def decompress_dir(path: str | Path) -> None:
    """
    Recursively decompresses all files in a directory.

    Args:
        path (str | Path): Path to parent directory.
    """
    path = Path(path)
    for parent, _, files in os.walk(path):
        for f in files:
            decompress_file(Path(parent, f))