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
def visit_Subscript(self, node):
    assert node.ctx.__class__.__name__ == 'Load'
    lhs = self.visit(node.value)
    slices = self.visit(node.slice)
    if _is_triton_tensor(lhs):
        return lhs.__getitem__(slices, _builder=self.builder)
    return lhs[slices]