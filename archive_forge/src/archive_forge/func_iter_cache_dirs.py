from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
def iter_cache_dirs(self) -> Iterator[str]:
    """:yield: all user and site cache directories."""
    yield self.user_cache_dir
    yield self.site_cache_dir