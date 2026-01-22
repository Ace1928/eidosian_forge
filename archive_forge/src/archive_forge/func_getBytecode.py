from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
def getBytecode(self) -> bytes:
    if not hasattr(self, 'bytecode'):
        self._assemble()
    return self.bytecode.tobytes()