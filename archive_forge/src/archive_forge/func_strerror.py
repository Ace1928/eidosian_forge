from ._cares import ffi as _ffi, lib as _lib
from .utils import maybe_str
def strerror(code):
    return maybe_str(_ffi.string(_lib.ares_strerror(code)))