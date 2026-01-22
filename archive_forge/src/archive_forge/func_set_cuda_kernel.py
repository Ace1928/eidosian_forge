import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def set_cuda_kernel(function):
    """
    Mark a function as a CUDA kernel. Kernels have the following requirements:

    - Metadata that marks them as a kernel.
    - Addition to the @llvm.used list, so that they will not be discarded.
    - The noinline attribute is not permitted, because this causes NVVM to emit
      a warning, which counts as failing IR verification.

    Presently it is assumed that there is one kernel per module, which holds
    for Numba-jitted functions. If this changes in future or this function is
    to be used externally, this function may need modification to add to the
    @llvm.used list rather than creating it.
    """
    module = function.module
    mdstr = ir.MetaDataString(module, 'kernel')
    mdvalue = ir.Constant(ir.IntType(32), 1)
    md = module.add_metadata((function, mdstr, mdvalue))
    nmd = cgutils.get_or_insert_named_metadata(module, 'nvvm.annotations')
    nmd.add(md)
    ptrty = ir.IntType(8).as_pointer()
    usedty = ir.ArrayType(ptrty, 1)
    fnptr = function.bitcast(ptrty)
    llvm_used = ir.GlobalVariable(module, usedty, 'llvm.used')
    llvm_used.linkage = 'appending'
    llvm_used.section = 'llvm.metadata'
    llvm_used.initializer = ir.Constant(usedty, [fnptr])
    function.attributes.discard('noinline')