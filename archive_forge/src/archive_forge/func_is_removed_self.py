from __future__ import annotations
import sys
from dataclasses import dataclass
from functools import reduce
import ctypes.wintypes  # noqa: E402
@property
def is_removed_self(self):
    return self.action == FILE_ACTION_REMOVED_SELF