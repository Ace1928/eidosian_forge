import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
def ast_unparse(node: ast.AST) -> T.Optional[str]:
    """Convert the AST node to source code as a string."""
    if hasattr(ast, 'unparse'):
        return ast.unparse(node)
    if isinstance(node, (ast.Str, ast.Num, ast.NameConstant, ast.Constant)):
        return str(ast_get_constant_value(node))
    if isinstance(node, ast.Name):
        return node.id
    return None