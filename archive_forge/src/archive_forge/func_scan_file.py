from __future__ import annotations
import json
import os
import pathlib
import pickle
import re
import sys
import typing as T
from ..backend.ninjabackend import ninja_quote
from ..compilers.compilers import lang_suffixes
def scan_file(self, fname: str) -> None:
    suffix = os.path.splitext(fname)[1][1:]
    if suffix != 'C':
        suffix = suffix.lower()
    if suffix in lang_suffixes['fortran']:
        self.scan_fortran_file(fname)
    elif suffix in lang_suffixes['cpp']:
        self.scan_cpp_file(fname)
    else:
        sys.exit(f'Can not scan files with suffix .{suffix}.')