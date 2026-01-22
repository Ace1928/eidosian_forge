import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes
def test_node_path_star(self):
    t = self._build_tree()
    self.assertEqual(10, len(find_all(t, '//*')))
    self.assertEqual(8, len(find_all(t, '//DefNode//*')))
    self.assertEqual(0, len(find_all(t, '//NameNode//*')))