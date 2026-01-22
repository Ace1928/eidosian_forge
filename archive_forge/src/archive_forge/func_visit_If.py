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
def visit_If(self, node):
    cond = self.visit(node.test)
    if _is_triton_tensor(cond):
        cond = cond.to(language.int1, _builder=self.builder)
        contains_return = ContainsReturnChecker(self.gscope).visit(node)
        if self.scf_stack and contains_return:
            raise UnsupportedLanguageConstruct(None, node, 'Cannot have `return` statements inside `while` or `for` statements in triton (note that this also applies to `return` statements that are inside functions transitively called from within `while`/`for` statements)')
        elif self.scf_stack or not contains_return:
            self.visit_if_scf(cond, node)
        else:
            self.visit_if_top_level(cond, node)
    else:
        cond = _unwrap_if_constexpr(cond)
        if type(cond) not in _condition_types:
            raise UnsupportedLanguageConstruct(None, node, '`if` conditionals can only accept values of type {{{}}}, not objects of type {}'.format(', '.join((_.__name__ for _ in _condition_types)), type(cond).__name__))
        if cond:
            self.visit_compound_statement(node.body)
        else:
            self.visit_compound_statement(node.orelse)