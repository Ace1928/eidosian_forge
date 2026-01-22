from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
def check_for_b902(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:

    def is_classmethod(decorators: Set[str]) -> bool:
        return any((name in decorators for name in self.b902_classmethod_decorators)) or node.name in B902.implicit_classmethods
    if len(self.contexts) < 2 or not isinstance(self.contexts[-2].node, ast.ClassDef):
        return
    cls = self.contexts[-2].node
    decorators: set[str] = {self.find_decorator_name(d) for d in node.decorator_list}
    if 'staticmethod' in decorators:
        return
    bases = {b.id for b in cls.bases if isinstance(b, ast.Name)}
    if any((basetype in bases for basetype in ('type', 'ABCMeta', 'EnumMeta'))):
        if is_classmethod(decorators):
            expected_first_args = B902.metacls
            kind = 'metaclass class'
        else:
            expected_first_args = B902.cls
            kind = 'metaclass instance'
    elif is_classmethod(decorators):
        expected_first_args = B902.cls
        kind = 'class'
    else:
        expected_first_args = B902.self
        kind = 'instance'
    args = getattr(node.args, 'posonlyargs', []) + node.args.args
    vararg = node.args.vararg
    kwarg = node.args.kwarg
    kwonlyargs = node.args.kwonlyargs
    if args:
        actual_first_arg = args[0].arg
        lineno = args[0].lineno
        col = args[0].col_offset
    elif vararg:
        actual_first_arg = '*' + vararg.arg
        lineno = vararg.lineno
        col = vararg.col_offset
    elif kwarg:
        actual_first_arg = '**' + kwarg.arg
        lineno = kwarg.lineno
        col = kwarg.col_offset
    elif kwonlyargs:
        actual_first_arg = '*, ' + kwonlyargs[0].arg
        lineno = kwonlyargs[0].lineno
        col = kwonlyargs[0].col_offset
    else:
        actual_first_arg = '(none)'
        lineno = node.lineno
        col = node.col_offset
    if actual_first_arg not in expected_first_args:
        if not actual_first_arg.startswith(('(', '*')):
            actual_first_arg = repr(actual_first_arg)
        self.errors.append(B902(lineno, col, vars=(actual_first_arg, kind, expected_first_args[0])))