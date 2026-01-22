import os
from breezy import errors, tests, workingtree
from breezy.mutabletree import BadReferenceTarget
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_root_present(self):
    """Subtree root is present, though not the working tree root"""
    tree, sub_tree = self.make_trees()
    if not tree.supports_setting_file_ids():
        self.skipTest('format does not support setting file ids')
    sub_tree.set_root_id(tree.path2id('file1'))
    try:
        self.assertRaises(BadReferenceTarget, tree.add_reference, sub_tree)
    except errors.UnsupportedOperation:
        self._references_unsupported(tree)