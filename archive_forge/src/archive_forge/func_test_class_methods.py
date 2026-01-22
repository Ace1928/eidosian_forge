from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.base import ast_utils
from pasta.base import scope
from pasta.base import test_utils
def test_class_methods(self):
    source = textwrap.dedent('        import aaa\n        class C:\n          def aaa(self):\n            return aaa\n\n          def bbb(self):\n            return aaa\n        ')
    tree = ast.parse(source)
    importstmt, classdef = tree.body
    method_aaa, method_bbb = classdef.body
    s = scope.analyze(tree)
    self.assertItemsEqual(s.names.keys(), {'aaa', 'C'})
    self.assertItemsEqual(s.external_references.keys(), {'aaa'})
    self.assertItemsEqual(s.names['aaa'].reads, [method_aaa.body[0].value, method_bbb.body[0].value])