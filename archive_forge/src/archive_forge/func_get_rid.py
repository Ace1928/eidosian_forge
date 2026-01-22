import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
def get_rid(cdata: FFI.CData) -> int:
    """Get the identifier for the R object.

    This is intended to be like Python's `id()`, but
    for R objects mapped by rpy2."""
    return int(ffi.cast('uintptr_t', cdata))