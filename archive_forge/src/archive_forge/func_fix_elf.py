from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def fix_elf(fname: str, rpath_dirs_to_remove: T.Set[bytes], new_rpath: T.Optional[bytes], verbose: bool=True) -> None:
    if new_rpath is not None:
        with Elf(fname, verbose) as e:
            e.fix_rpath(fname, rpath_dirs_to_remove, new_rpath)