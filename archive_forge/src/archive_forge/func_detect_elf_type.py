from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def detect_elf_type(self) -> T.Tuple[int, bool]:
    data = self.bf.read(6)
    if data[1:4] != b'ELF':
        if self.verbose:
            print(f'File {self.bfile!r} is not an ELF file.')
        sys.exit(0)
    if data[4] == 1:
        ptrsize = 32
    elif data[4] == 2:
        ptrsize = 64
    else:
        sys.exit(f'File {self.bfile!r} has unknown ELF class.')
    if data[5] == 1:
        is_le = True
    elif data[5] == 2:
        is_le = False
    else:
        sys.exit(f'File {self.bfile!r} has unknown ELF endianness.')
    return (ptrsize, is_le)