from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def open_bf(self, bfile: str) -> None:
    self.bf = None
    self.bf_perms = None
    try:
        self.bf = open(bfile, 'r+b')
    except PermissionError as e:
        self.bf_perms = stat.S_IMODE(os.lstat(bfile).st_mode)
        os.chmod(bfile, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        try:
            self.bf = open(bfile, 'r+b')
        except Exception:
            os.chmod(bfile, self.bf_perms)
            self.bf_perms = None
            raise e