import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes
def test_node_path_node_predicate_step(self):
    t = self._build_tree()
    self.assertEqual(2, len(find_all(t, '//DefNode[.//NameNode]')))
    self.assertEqual(8, len(find_all(t, '//DefNode[.//NameNode]//*')))
    self.assertEqual(1, len(find_all(t, '//DefNode[.//NameNode]//ReturnStatNode')))
    self.assertEqual(Nodes.ReturnStatNode, type(find_first(t, '//DefNode[.//NameNode]//ReturnStatNode')))