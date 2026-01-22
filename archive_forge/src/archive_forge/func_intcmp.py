from importlib import import_module
import os
import sys
def intcmp(s1, s2):
    try:
        _i1 = int(s1)
        _i2 = int(s2)
    except ValueError:
        _i1 = s1
        _i2 = s2
    if _i1 < _i2:
        return -1
    if _i1 > _i2:
        return 1
    else:
        return 0