from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
def fromAssembly(self, assembly: List[str] | str) -> None:
    if isinstance(assembly, list):
        self.assembly = assembly
    elif isinstance(assembly, str):
        self.assembly = assembly.splitlines()
    else:
        raise TypeError(f'expected str or List[str], got {type(assembly).__name__}')
    if hasattr(self, 'bytecode'):
        del self.bytecode