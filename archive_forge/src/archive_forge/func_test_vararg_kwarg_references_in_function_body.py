from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_vararg_kwarg_references_in_function_body(self):
    source = textwrap.dedent('        def aaa(bbb, *ccc, **ddd):\n          ccc\n          ddd\n        eee(ccc, ddd)\n        ')
    tree = ast.parse(source)
    funcdef, call = tree.body
    ccc_expr, ddd_expr = funcdef.body
    sc = scope.analyze(tree)
    func_scope = sc.lookup_scope(funcdef)
    self.assertIn('ccc', func_scope.names)
    self.assertItemsEqual(func_scope.names['ccc'].reads, [ccc_expr.value])
    self.assertIn('ddd', func_scope.names)
    self.assertItemsEqual(func_scope.names['ddd'].reads, [ddd_expr.value])