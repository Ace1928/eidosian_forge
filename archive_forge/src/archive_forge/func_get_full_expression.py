import ast
import sys
from typing import Any
from .internal import Filters, Key
def get_full_expression(self, node):
    if isinstance(node, ast.Attribute):
        return self.get_full_expression(node.value) + '.' + node.attr
    elif isinstance(node, ast.Name):
        return node.id
    else:
        return 'ArbitraryExpression'