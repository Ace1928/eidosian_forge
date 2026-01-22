import os
import warnings
import re
def parsefield(line, i, n):
    """Separate one key-value pair in a mailcap entry."""
    start = i
    while i < n:
        c = line[i]
        if c == ';':
            break
        elif c == '\\':
            i = i + 2
        else:
            i = i + 1
    return (line[start:i].strip(), i)