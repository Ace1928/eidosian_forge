from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def fix_rpath(self, fname: str, rpath_dirs_to_remove: T.Set[bytes], new_rpath: bytes) -> None:
    self.fix_rpathtype_entry(fname, rpath_dirs_to_remove, new_rpath, DT_RPATH)
    self.fix_rpathtype_entry(fname, rpath_dirs_to_remove, new_rpath, DT_RUNPATH)