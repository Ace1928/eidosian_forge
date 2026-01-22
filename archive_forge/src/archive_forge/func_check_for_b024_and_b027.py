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
def check_for_b024_and_b027(self, node: ast.ClassDef):
    """Check for inheritance from abstract classes in abc and lack of
        any methods decorated with abstract*"""

    def is_abc_class(value, name='ABC'):
        if isinstance(value, ast.keyword):
            return value.arg == 'metaclass' and is_abc_class(value.value, 'ABCMeta')
        return isinstance(value, ast.Name) and value.id == name or (isinstance(value, ast.Attribute) and value.attr == name and isinstance(value.value, ast.Name) and (value.value.id == 'abc'))

    def is_abstract_decorator(expr):
        return isinstance(expr, ast.Name) and expr.id[:8] == 'abstract' or (isinstance(expr, ast.Attribute) and expr.attr[:8] == 'abstract')

    def is_overload(expr):
        return isinstance(expr, ast.Name) and expr.id == 'overload' or (isinstance(expr, ast.Attribute) and expr.attr == 'overload')

    def empty_body(body) -> bool:

        def is_str_or_ellipsis(node):
            return isinstance(node, ast.Constant) and (node.value is Ellipsis or isinstance(node.value, str))
        return all((isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and is_str_or_ellipsis(stmt.value)) for stmt in body))
    if len(node.bases) + len(node.keywords) > 1:
        return
    if not any(map(is_abc_class, (*node.bases, *node.keywords))):
        return
    has_method = False
    has_abstract_method = False
    for stmt in node.body:
        if isinstance(stmt, (ast.AnnAssign, ast.Assign)):
            has_abstract_method = True
            continue
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        has_method = True
        has_abstract_decorator = any(map(is_abstract_decorator, stmt.decorator_list))
        has_abstract_method |= has_abstract_decorator
        if not has_abstract_decorator and empty_body(stmt.body) and (not any(map(is_overload, stmt.decorator_list))):
            self.errors.append(B027(stmt.lineno, stmt.col_offset, vars=(stmt.name,)))
    if has_method and (not has_abstract_method):
        self.errors.append(B024(node.lineno, node.col_offset, vars=(node.name,)))