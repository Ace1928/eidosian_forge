import ctypes
import warnings
from .base import _LIB, check_call, c_str, ProfileHandle, c_str_array, py_str, KVStoreHandle
def dump_profile():
    """Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally."""
    warnings.warn('profiler.dump_profile() is deprecated. Please use profiler.dump() instead')
    dump(True)