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
def is_cdata_sexp(obj: typing.Any) -> bool:
    """Is the object a cffi `CData` object pointing to an R object ?"""
    if isinstance(obj, FFI.CData) and ffi.typeof(obj).cname == 'struct SEXPREC *':
        return True
    else:
        return False