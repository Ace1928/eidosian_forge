import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def check_jit_execution():
    """
    Check the system allows execution of in-memory JITted functions.
    An exception is raised otherwise.
    """
    errno = ffi.lib.LLVMPY_TryAllocateExecutableMemory()
    if errno != 0:
        raise OSError(errno, 'cannot allocate executable memory. This may be due to security restrictions on your system, such as SELinux or similar mechanisms.')