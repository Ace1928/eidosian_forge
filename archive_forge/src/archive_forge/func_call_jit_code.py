from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def call_jit_code(self, func, sig, args):
    """Calls into Numba jitted code and propagate error using the Python
        calling convention.

        Parameters
        ----------
        func : function
            The Python function to be compiled. This function is compiled
            in nopython-mode.
        sig : numba.typing.Signature
            The function signature for *func*.
        args : Sequence[llvmlite.binding.Value]
            LLVM values to use as arguments.

        Returns
        -------
        (is_error, res) :  2-tuple of llvmlite.binding.Value.
            is_error : true iff *func* raised an exception.
            res : Returned value from *func* iff *is_error* is false.

        If *is_error* is true, this method will adapt the nopython exception
        into a Python exception. Caller should return NULL to Python to
        indicate an error.
        """
    builder = self.builder
    cres = self.context.compile_subroutine(builder, func, sig)
    got_retty = cres.signature.return_type
    retty = sig.return_type
    if got_retty != retty:
        raise errors.LoweringError(f'mismatching signature {got_retty} != {retty}.\n')
    status, res = self.context.call_internal_no_propagate(builder, cres.fndesc, sig, args)
    is_error_ptr = cgutils.alloca_once(builder, cgutils.bool_t, zfill=True)
    res_type = self.context.get_value_type(sig.return_type)
    res_ptr = cgutils.alloca_once(builder, res_type, zfill=True)
    with builder.if_else(status.is_error) as (has_err, no_err):
        with has_err:
            builder.store(status.is_error, is_error_ptr)
            self.context.call_conv.raise_error(builder, self, status)
        with no_err:
            res = imputils.fix_returning_optional(self.context, builder, sig, status, res)
            builder.store(res, res_ptr)
    is_error = builder.load(is_error_ptr)
    res = builder.load(res_ptr)
    return (is_error, res)