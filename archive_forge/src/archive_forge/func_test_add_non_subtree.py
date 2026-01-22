import os
from breezy import errors, tests, workingtree
from breezy.mutabletree import BadReferenceTarget
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_non_subtree(self):
    tree, sub_tree = self.make_trees()
    os.rename('tree/sub-tree', 'sibling')
    sibling = workingtree.WorkingTree.open('sibling')
    try:
        self.assertRaises(BadReferenceTarget, tree.add_reference, sibling)
    except errors.UnsupportedOperation:
        self._references_unsupported(tree)