import ast
import inspect
import re
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from .. import language
from .._C.libtriton.triton import ir
from ..language import constexpr, tensor
from ..runtime import JITFunction
from .errors import (CompilationError, CompileTimeAssertionFailure, UnsupportedLanguageConstruct)
def mangle_ty(ty):
    if ty.is_ptr():
        return 'P' + mangle_ty(ty.element_ty)
    if ty.is_int():
        SIGNED = language.dtype.SIGNEDNESS.SIGNED
        prefix = 'i' if ty.int_signedness == SIGNED else 'u'
        return prefix + str(ty.int_bitwidth)
    if ty.is_floating():
        return str(ty)
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = '_'.join(map(str, ty.shape))
        return f'{elt}S{shape}S'
    if ty.is_void():
        return 'V'
    assert False, 'Unsupported type'