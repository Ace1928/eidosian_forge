import os
from breezy import errors, tests, workingtree
from breezy.mutabletree import BadReferenceTarget
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_reference_same_root(self):
    tree = self.make_branch_and_tree('tree')
    if not tree.supports_setting_file_ids():
        self.skipTest('format does not support setting file ids')
    self.build_tree(['tree/file1'])
    tree.add('file1')
    tree.set_root_id(b'root-id')
    sub_tree = self.make_branch_and_tree('tree/sub-tree')
    sub_tree.set_root_id(b'root-id')
    try:
        self.assertRaises(BadReferenceTarget, tree.add_reference, sub_tree)
    except errors.UnsupportedOperation:
        self._references_unsupported(tree)