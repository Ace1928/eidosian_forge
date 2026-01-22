from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
def char_info(c):
    return 4 * (char_base + c)