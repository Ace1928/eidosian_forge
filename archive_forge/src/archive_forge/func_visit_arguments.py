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
def visit_arguments(self, node):
    arg_names = []
    for arg in node.args:
        arg_names += [self.visit(arg)]
    kwarg_names = self.visit(node.kwarg)
    return (arg_names, kwarg_names)