from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def get_runpath(self) -> T.Optional[str]:
    offset = self.get_entry_offset(DT_RUNPATH)
    if offset is None:
        return None
    self.bf.seek(offset)
    return self.read_str().decode()