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
def execute_static_assert(self, node: ast.Call) -> None:
    arg_count = len(node.args)
    if not 0 < arg_count <= 2 or len(node.keywords):
        raise TypeError('`static_assert` requires one or two positional arguments only')
    passed = _unwrap_if_constexpr(self.visit(node.args[0]))
    if not isinstance(passed, bool):
        raise NotImplementedError('Assertion condition could not be determined at compile-time. Make sure that it depends only on `constexpr` values')
    if not passed:
        if arg_count == 1:
            message = ''
        else:
            try:
                message = self.visit(node.args[1])
            except Exception as e:
                message = '<failed to evaluate assertion message: ' + repr(e) + '>'
        raise CompileTimeAssertionFailure(None, node, _unwrap_if_constexpr(message))
    return None