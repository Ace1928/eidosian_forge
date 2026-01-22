import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def has_conflicts(self) -> bool:
    for value in self._byname.values():
        if isinstance(value, ConflictedIndexEntry):
            return True
    return False