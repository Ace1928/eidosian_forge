from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def hexStr(s):
    h = string.hexdigits
    r = ''
    for c in s:
        i = byteord(c)
        r = r + h[i >> 4 & 15] + h[i & 15]
    return r