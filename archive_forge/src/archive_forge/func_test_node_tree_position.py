import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_tree_position(self):
    """
        Test matching on nodes based on NLTK tree position.
        """
    tree = ParentedTree.fromstring('(S (NP-SBJ x) (NP x) (NNP x) (VP x))')
    leaf_positions = {tree.leaf_treeposition(x) for x in range(len(tree.leaves()))}
    tree_positions = [x for x in tree.treepositions() if x not in leaf_positions]
    for position in tree_positions:
        node_id = f'N{position}'
        tgrep_positions = list(tgrep.tgrep_positions(node_id, [tree]))
        self.assertEqual(len(tgrep_positions[0]), 1)
        self.assertEqual(tgrep_positions[0][0], position)