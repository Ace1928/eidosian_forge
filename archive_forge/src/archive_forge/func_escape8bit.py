from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
def escape8bit(data):
    """Input is Unicode string."""

    def escapechar(c):
        n = ord(c)
        if 32 <= n <= 127 and c not in '<&>':
            return c
        else:
            return '&#' + repr(n) + ';'
    return strjoin(map(escapechar, data.decode('latin-1')))