from __future__ import annotations
import sys
import os
import stat
import struct
import shutil
import subprocess
import typing as T
from ..mesonlib import OrderedSet, generate_list, Popen_safe
def get_darwin_rpaths_to_remove(fname: str) -> T.List[str]:
    p, out, _ = Popen_safe(['otool', '-l', fname], stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args, out)
    result = []
    current_cmd = 'FOOBAR'
    for line in out.split('\n'):
        line = line.strip()
        if ' ' not in line:
            continue
        key, value = line.strip().split(' ', 1)
        if key == 'cmd':
            current_cmd = value
        if key == 'path' and current_cmd == 'LC_RPATH':
            rp = value.split('(', 1)[0].strip()
            result.append(rp)
    return result