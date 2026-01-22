from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def do_hexstring(self, token):
    hexStr = ''.join(token[1:-1].split())
    if len(hexStr) % 2:
        hexStr = hexStr + '0'
    cleanstr = []
    for i in range(0, len(hexStr), 2):
        cleanstr.append(chr(int(hexStr[i:i + 2], 16)))
    cleanstr = ''.join(cleanstr)
    return ps_string(cleanstr)