from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def remove_rpath_entry(self, entrynum: int) -> None:
    sec = self.find_section(b'.dynamic')
    if sec is None:
        return None
    for i, entry in enumerate(self.dynamic):
        if entry.d_tag == entrynum:
            rpentry = self.dynamic[i]
            rpentry.d_tag = 0
            self.dynamic = self.dynamic[:i] + self.dynamic[i + 1:] + [rpentry]
            break
    for entry in self.dynamic[i:]:
        if entry.d_tag == DT_MIPS_RLD_MAP_REL:
            entry.val += 2 * (self.ptrsize // 8)
            break
    self.bf.seek(sec.sh_offset)
    for entry in self.dynamic:
        entry.write(self.bf)
    return None