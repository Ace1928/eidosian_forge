from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
def visit_in_order(self, node, *attrs):
    for attr in attrs:
        val = getattr(node, attr, None)
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                self.visit(item)
        elif isinstance(val, ast.AST):
            self.visit(val)