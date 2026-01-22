from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_functiondef_nested_imports(self):
    source = textwrap.dedent('        def foo(bar):\n          import aaa\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    node_aaa = ast_utils.find_nodes_by_type(tree, ast.alias)[0]
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})