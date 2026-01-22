import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
def find_c_enchant_lib():
    verbose = os.environ.get('PYENCHANT_VERBOSE_FIND')
    if verbose:
        global VERBOSE_FIND
        VERBOSE_FIND = True
    prefix = os.environ.get('PYENCHANT_ENCHANT_PREFIX')
    if prefix:
        return from_prefix(prefix)
    library_path = os.environ.get('PYENCHANT_LIBRARY_PATH')
    if library_path:
        return from_env_var(library_path)
    from_package = from_package_resources()
    if from_package:
        return from_package
    return from_system()