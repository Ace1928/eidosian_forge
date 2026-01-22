import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir
from numba.core import (cgutils, config, debuginfo, itanium_mangler, types,
from numba.core.dispatcher import Dispatcher
from numba.core.base import BaseContext
from numba.core.callconv import BaseCallConv, MinimalCallConv
from numba.core.typing import cmathdecl
from numba.core import datamodel
from .cudadrv import nvvm
from numba.cuda import codegen, nvvmutils, ufuncs
from numba.cuda.models import cuda_data_manager
def make_constant_array(self, builder, aryty, arr):
    """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """
    lmod = builder.module
    constvals = [self.get_constant(types.byte, i) for i in iter(arr.tobytes(order='A'))]
    constaryty = ir.ArrayType(ir.IntType(8), len(constvals))
    constary = ir.Constant(constaryty, constvals)
    addrspace = nvvm.ADDRSPACE_CONSTANT
    gv = cgutils.add_global_variable(lmod, constary.type, '_cudapy_cmem', addrspace=addrspace)
    gv.linkage = 'internal'
    gv.global_constant = True
    gv.initializer = constary
    lldtype = self.get_data_type(aryty.dtype)
    align = self.get_abi_sizeof(lldtype)
    gv.align = 2 ** (align - 1).bit_length()
    ptrty = ir.PointerType(ir.IntType(8))
    genptr = builder.addrspacecast(gv, ptrty, 'generic')
    ary = self.make_array(aryty)(self, builder)
    kshape = [self.get_constant(types.intp, s) for s in arr.shape]
    kstrides = [self.get_constant(types.intp, s) for s in arr.strides]
    self.populate_array(ary, data=builder.bitcast(genptr, ary.data.type), shape=kshape, strides=kstrides, itemsize=ary.itemsize, parent=ary.parent, meminfo=None)
    return ary._getvalue()