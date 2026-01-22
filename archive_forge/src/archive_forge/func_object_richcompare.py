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
def object_richcompare(self, lhs, rhs, opstr):
    """
        Refer to Python source Include/object.h for macros definition
        of the opid.
        """
    ops = ['<', '<=', '==', '!=', '>', '>=']
    if opstr in ops:
        opid = ops.index(opstr)
        fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj, ir.IntType(32)])
        fn = self._get_function(fnty, name='PyObject_RichCompare')
        lopid = self.context.get_constant(types.int32, opid)
        return self.builder.call(fn, (lhs, rhs, lopid))
    elif opstr == 'is':
        bitflag = self.builder.icmp_unsigned('==', lhs, rhs)
        return self.bool_from_bool(bitflag)
    elif opstr == 'is not':
        bitflag = self.builder.icmp_unsigned('!=', lhs, rhs)
        return self.bool_from_bool(bitflag)
    elif opstr in ('in', 'not in'):
        fnty = ir.FunctionType(ir.IntType(32), [self.pyobj, self.pyobj])
        fn = self._get_function(fnty, name='PySequence_Contains')
        status = self.builder.call(fn, (rhs, lhs))
        negone = self.context.get_constant(types.int32, -1)
        is_good = self.builder.icmp_unsigned('!=', status, negone)
        outptr = cgutils.alloca_once_value(self.builder, Constant(self.pyobj, None))
        with cgutils.if_likely(self.builder, is_good):
            if opstr == 'not in':
                status = self.builder.not_(status)
            truncated = self.builder.trunc(status, ir.IntType(1))
            self.builder.store(self.bool_from_bool(truncated), outptr)
        return self.builder.load(outptr)
    else:
        raise NotImplementedError('Unknown operator {op!r}'.format(op=opstr))