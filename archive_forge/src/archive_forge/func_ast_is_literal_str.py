import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
def ast_is_literal_str(node: ast.AST) -> bool:
    """Return True if the given node is a literal string."""
    return isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.Str)) and isinstance(ast_get_constant_value(node.value), str)