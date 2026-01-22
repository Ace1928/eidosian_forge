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
def visit_Return(self, node):
    ret_value = self.visit(node.value)
    if ret_value is None:
        self.builder.ret([])
        ret_ty = None
    elif isinstance(ret_value, tuple):
        ret_values = [language.core._to_tensor(v, self.builder) for v in ret_value]
        ret_types = [v.type for v in ret_values]
        self.builder.ret([v.handle for v in ret_values])
        ret_ty = tuple(ret_types)
    else:
        ret = language.core._to_tensor(ret_value, self.builder)
        self.builder.ret([ret.handle])
        ret_ty = ret.type
    return ret_ty