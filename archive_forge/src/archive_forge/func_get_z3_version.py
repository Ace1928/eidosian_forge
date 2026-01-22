import common_z3 as CM_Z3
import ctypes
from .z3 import *
def get_z3_version(as_str=False):
    major = ctypes.c_uint(0)
    minor = ctypes.c_uint(0)
    build = ctypes.c_uint(0)
    rev = ctypes.c_uint(0)
    Z3_get_version(major, minor, build, rev)
    rs = map(int, (major.value, minor.value, build.value, rev.value))
    if as_str:
        return '{}.{}.{}.{}'.format(*rs)
    else:
        return rs