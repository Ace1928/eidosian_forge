import unittest
from collections import Counter
from low_index import *
def test_permutation_rep(self):
    node = SimsNode(2, 2)
    node.add_edge(1, 1, 1)
    node.add_edge(2, 1, 2)
    node.add_edge(2, 2, 1)
    node.add_edge(1, 2, 2)
    self.assertTrue(node.is_complete())
    self.assertEqual(node.degree, 2)
    self.assertEqual(node.permutation_rep(), [[0, 1], [1, 0]])