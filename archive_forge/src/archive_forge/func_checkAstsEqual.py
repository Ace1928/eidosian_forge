from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import sys
import unittest
from six.moves import zip
def checkAstsEqual(self, a, b):
    """Compares two ASTs and fails if there are differences.

    Ignores `ctx` fields and formatting info.
    """
    if a is None and b is None:
        return
    try:
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        for node_a, node_b in zip(ast.walk(a), ast.walk(b)):
            self.assertEqual(type(node_a), type(node_b))
            for field in type(node_a)()._fields:
                a_val = getattr(node_a, field, None)
                b_val = getattr(node_b, field, None)
                if isinstance(a_val, list):
                    for item_a, item_b in zip(a_val, b_val):
                        self.checkAstsEqual(item_a, item_b)
                elif isinstance(a_val, ast.AST) or isinstance(b_val, ast.AST):
                    if not isinstance(a_val, (ast.Load, ast.Store, ast.Param)) and (not isinstance(b_val, (ast.Load, ast.Store, ast.Param))):
                        self.assertIsNotNone(a_val)
                        self.assertIsNotNone(b_val)
                        self.checkAstsEqual(a_val, b_val)
                else:
                    self.assertEqual(a_val, b_val)
    except AssertionError as ae:
        self.fail('ASTs differ:\n%s\n  !=\n%s\n\n%s' % (ast.dump(a), ast.dump(b), ae))