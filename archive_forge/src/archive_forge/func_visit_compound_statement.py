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
def visit_compound_statement(self, stmts):
    if not _is_list_like(stmts):
        stmts = [stmts]
    for stmt in stmts:
        ret_type = self.visit(stmt)
        if ret_type is not None and isinstance(stmt, ast.Return):
            self.last_ret_type = ret_type