import ast
import inspect
import sys
import textwrap
import typing as T
from types import ModuleType
from .common import Docstring, DocstringParam
def ast_get_attribute(node: ast.AST) -> T.Optional[T.Tuple[str, T.Optional[str], T.Optional[str]]]:
    """Return name, type and default if the given node is an attribute."""
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if isinstance(target, ast.Name):
            type_str = None
            if isinstance(node, ast.AnnAssign):
                type_str = ast_unparse(node.annotation)
            default = None
            if node.value:
                default = ast_unparse(node.value)
            return (target.id, type_str, default)
    return None