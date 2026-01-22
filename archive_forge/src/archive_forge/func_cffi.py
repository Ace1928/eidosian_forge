import ctypes
from functools import cached_property
from numba.core import compiler, registry
from numba.core.caching import NullCache, FunctionCache
from numba.core.dispatcher import _FunctionCompiler
from numba.core.typing import signature
from numba.core.typing.ctypes_utils import to_ctypes
from numba.core.compiler_lock import global_compiler_lock
@cached_property
def cffi(self):
    """
        A cffi function pointer representing the C callback.
        """
    import cffi
    ffi = cffi.FFI()
    return ffi.cast('void *', self.address)