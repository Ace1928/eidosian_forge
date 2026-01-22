from typing import Collection, List
from sys import maxsize
def leading_white_space(s: str) -> int:
    i = 0
    for c in s:
        if c not in ' \t':
            return i
        i += 1
    return i