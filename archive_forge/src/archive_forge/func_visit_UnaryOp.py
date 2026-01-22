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
def visit_UnaryOp(self, node):
    op = self.visit(node.operand)
    fn = self._method_name_for_unary_op.get(type(node.op))
    if fn is None:
        raise UnsupportedLanguageConstruct(None, node, "AST unary operator '{}' is not (currently) implemented.".format(node.op.__name__))
    if _is_triton_tensor(op):
        return getattr(op, fn)(_builder=self.builder)
    return getattr(op, fn)()