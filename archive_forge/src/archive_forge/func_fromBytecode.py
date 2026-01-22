from __future__ import annotations
from fontTools.misc.textTools import num2binary, binary2num, readHex, strjoin
import array
from io import StringIO
from typing import List
import re
import logging
def fromBytecode(self, bytecode: bytes) -> None:
    self.bytecode = array.array('B', bytecode)
    if hasattr(self, 'assembly'):
        del self.assembly