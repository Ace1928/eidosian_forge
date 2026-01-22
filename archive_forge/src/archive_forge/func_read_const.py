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
def read_const(self, index):
    """
        Look up constant number *index* inside the environment body.
        A borrowed reference is returned.

        The returned LLVM value may have NULL value at runtime which indicates
        an error at runtime.
        """
    assert index < len(self.env.consts)
    builder = self.pyapi.builder
    consts = self.env_body.consts
    ret = cgutils.alloca_once(builder, self.pyapi.pyobj, zfill=True)
    with builder.if_else(cgutils.is_not_null(builder, consts)) as (br_not_null, br_null):
        with br_not_null:
            getitem = self.pyapi.list_getitem(consts, index)
            builder.store(getitem, ret)
        with br_null:
            self.pyapi.err_set_string('PyExc_RuntimeError', '`env.consts` is NULL in `read_const`')
    return builder.load(ret)