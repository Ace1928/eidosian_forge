from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def get_entry_offset(self, entrynum: int) -> T.Optional[int]:
    sec = self.find_section(b'.dynstr')
    for i in self.dynamic:
        if i.d_tag == entrynum:
            res = sec.sh_offset + i.val
            assert isinstance(res, int)
            return res
    return None