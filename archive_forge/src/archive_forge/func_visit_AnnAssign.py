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
def visit_AnnAssign(self, node):
    annotation = self.visit(node.annotation)
    target = self.visit(node.target)
    value = self.visit(node.value)
    if annotation == constexpr:
        if target in self.lscope:
            raise ValueError(f'{target} is already defined. constexpr cannot be reassigned.')
        if not _is_constexpr(value):
            value = constexpr(value)
        self.lscope[target] = value
        return self.lscope[target]
    return self.visit_Assign(node)