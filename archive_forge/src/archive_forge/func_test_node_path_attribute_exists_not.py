import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.TreePath import find_first, find_all
from Cython.Compiler import Nodes, ExprNodes
def test_node_path_attribute_exists_not(self):
    t = self._build_tree()
    self.assertEqual(0, len(find_all(t, '//NameNode[not(@name)]')))
    self.assertEqual(2, len(find_all(t, '//NameNode[not(@honking)]')))