from __future__ import absolute_import
import types
from . import Errors
def chars_to_ranges(s):
    """
    Return a list of character codes consisting of pairs
    [code1a, code1b, code2a, code2b,...] which cover all
    the characters in |s|.
    """
    char_list = list(s)
    char_list.sort()
    i = 0
    n = len(char_list)
    result = []
    while i < n:
        code1 = ord(char_list[i])
        code2 = code1 + 1
        i += 1
        while i < n and code2 >= ord(char_list[i]):
            code2 += 1
            i += 1
        result.append(code1)
        result.append(code2)
    return result