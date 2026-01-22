import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
def test_wrap_singlestat(self):
    t = self.run_pipeline([NormalizeTree(None)], u'if x: y')
    self.assertLines(u'\n(root): StatListNode\n  stats[0]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: StatListNode\n        stats[0]: ExprStatNode\n          expr: NameNode\n', self.treetypes(t))