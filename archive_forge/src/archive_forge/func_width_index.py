from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
def width_index(c):
    return data[char_info(c)]