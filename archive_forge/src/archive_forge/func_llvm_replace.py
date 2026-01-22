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
def llvm_replace(llvmir):
    replacements = [('declare double @"___numba_atomic_double_add"(double* %".1", double %".2")', ir_numba_atomic_binary(T='double', Ti='i64', OP='fadd', FUNC='add')), ('declare float @"___numba_atomic_float_sub"(float* %".1", float %".2")', ir_numba_atomic_binary(T='float', Ti='i32', OP='fsub', FUNC='sub')), ('declare double @"___numba_atomic_double_sub"(double* %".1", double %".2")', ir_numba_atomic_binary(T='double', Ti='i64', OP='fsub', FUNC='sub')), ('declare i64 @"___numba_atomic_u64_inc"(i64* %".1", i64 %".2")', ir_numba_atomic_inc(T='i64', Tu='u64')), ('declare i64 @"___numba_atomic_u64_dec"(i64* %".1", i64 %".2")', ir_numba_atomic_dec(T='i64', Tu='u64')), ('declare float @"___numba_atomic_float_max"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='', OP='nnan olt', PTR_OR_VAL='ptr', FUNC='max')), ('declare double @"___numba_atomic_double_max"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='', OP='nnan olt', PTR_OR_VAL='ptr', FUNC='max')), ('declare float @"___numba_atomic_float_min"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='', OP='nnan ogt', PTR_OR_VAL='ptr', FUNC='min')), ('declare double @"___numba_atomic_double_min"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='', OP='nnan ogt', PTR_OR_VAL='ptr', FUNC='min')), ('declare float @"___numba_atomic_float_nanmax"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='nan', OP='ult', PTR_OR_VAL='', FUNC='max')), ('declare double @"___numba_atomic_double_nanmax"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='nan', OP='ult', PTR_OR_VAL='', FUNC='max')), ('declare float @"___numba_atomic_float_nanmin"(float* %".1", float %".2")', ir_numba_atomic_minmax(T='float', Ti='i32', NAN='nan', OP='ugt', PTR_OR_VAL='', FUNC='min')), ('declare double @"___numba_atomic_double_nanmin"(double* %".1", double %".2")', ir_numba_atomic_minmax(T='double', Ti='i64', NAN='nan', OP='ugt', PTR_OR_VAL='', FUNC='min')), ('immarg', '')]
    for decl, fn in replacements:
        llvmir = llvmir.replace(decl, fn)
    llvmir = llvm140_to_70_ir(llvmir)
    return llvmir