import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
@ffi.def_extern()
def raw_free(o: object) -> None:
    if o in _allocated:
        _allocated.remove(o)