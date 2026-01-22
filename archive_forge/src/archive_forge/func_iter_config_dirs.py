from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
def iter_config_dirs(self) -> Iterator[str]:
    """:yield: all user and site configuration directories."""
    yield self.user_config_dir
    yield self.site_config_dir