import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def intrin(context, x):
    sig = signature(types.intp, x)
    if isinstance(x, types.IntegerLiteral):
        if x.literal_value == 1:

            def codegen(context, builder, signature, args):
                atype = signature.args[0]
                llrtype = context.get_value_type(atype)
                return ir.Constant(llrtype, 51966)
            return (sig, codegen)
        else:
            raise errors.TypingError('literal value')
    else:

        def codegen(context, builder, signature, args):
            atype = signature.return_type
            llrtype = context.get_value_type(atype)
            int_100 = ir.Constant(llrtype, 100)
            return builder.mul(args[0], int_100)
        return (sig, codegen)