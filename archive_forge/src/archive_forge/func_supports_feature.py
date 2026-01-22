from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import sys
import unittest
from six.moves import zip
def supports_feature(feature):
    if feature == 'bytes_node':
        return hasattr(ast, 'Bytes') and issubclass(ast.Bytes, ast.AST)
    if feature == 'exec_node':
        return hasattr(ast, 'Exec') and issubclass(ast.Exec, ast.AST)
    if feature == 'type_annotations':
        try:
            ast.parse('def foo(bar: str=123) -> None: pass')
        except SyntaxError:
            return False
        return True
    if feature == 'fstring':
        return hasattr(ast, 'JoinedStr') and issubclass(ast.JoinedStr, ast.AST)
    if feature == 'mixed_tabs_spaces':
        return sys.version_info[0] < 3
    return False