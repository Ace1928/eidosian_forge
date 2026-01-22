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
Calls into Numba jitted code and propagate error using the Python
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
        