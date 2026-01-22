from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
def iter_data_paths(self) -> Iterator[Path]:
    """:yield: all user and site data paths."""
    for path in self.iter_data_dirs():
        yield Path(path)