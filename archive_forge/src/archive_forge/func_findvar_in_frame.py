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
def findvar_in_frame(rho, symbol):
    """Safer wrapper around Rf_findVarInFrame()

    Run the function Rf_findVarInFrame() in R's C-API through
    R_ToplevelExec().

    Note: All arguments, and the object returned, are C-level
    R objects.

    Args:
    - rho: An R environment.
    - symbol: An R symbol (as returned by Rf_install())
    Returns:
    The object found.
    """
    exec_data = ffi.new('struct RPY2_sym_env_data *', [symbol, rho, openrlib.rlib.R_NilValue])
    _ = openrlib.rlib.R_ToplevelExec(openrlib.rlib._exec_findvar_in_frame, exec_data)
    if _ != openrlib.rlib.TRUE:
        raise embedded.RRuntimeError('R C-API Rf_findVarInFrame()')
    return exec_data.data