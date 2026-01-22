from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_classdef_nested_imports(self):
    source = textwrap.dedent('        class Foo():\n          import aaa\n        ')
    tree = ast.parse(source)
    nodes = tree.body
    node_aaa = nodes[0].body[0].names[0]
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'Foo'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})