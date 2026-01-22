import os
import shutil
from breezy import errors, mutabletree, tests
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.osutils import supports_symlinks
from breezy.tests import features
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_intertree import TestCaseWithTwoTrees
from breezy.tree import TreeChange
def test_unchanged_unicode(self):
    tree1 = self.make_branch_and_tree('tree1')
    tree2 = self.make_to_branch_and_tree('tree2')
    tree2.set_root_id(tree1.path2id(''))
    a_id = 'α-id'.encode()
    subfile_id = 'ω-subfile-id'.encode()
    rootfile_id = 'ω-root-id'.encode()
    try:
        self.build_tree(['tree1/α/', 'tree2/α/'])
    except UnicodeError:
        raise tests.TestSkipped('Could not create Unicode files.')
    self.build_tree_contents([('tree1/α/ω-subfile', b'sub contents\n'), ('tree2/α/ω-subfile', b'sub contents\n'), ('tree1/ω-rootfile', b'root contents\n'), ('tree2/ω-rootfile', b'root contents\n')])
    tree1.add(['α', 'α/ω-subfile', 'ω-rootfile'], ids=[a_id, subfile_id, rootfile_id])
    tree2.add(['α', 'α/ω-subfile', 'ω-rootfile'], ids=[a_id, subfile_id, rootfile_id])
    tree1, tree2 = self.mutable_trees_to_locked_test_trees(tree1, tree2)
    expected = self.sorted([self.unchanged(tree1, ''), self.unchanged(tree1, 'α'), self.unchanged(tree1, 'α/ω-subfile'), self.unchanged(tree1, 'ω-rootfile')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, include_unchanged=True))
    expected = self.sorted([self.unchanged(tree1, 'α'), self.unchanged(tree1, 'α/ω-subfile')])
    self.assertEqual(expected, self.do_iter_changes(tree1, tree2, specific_files=['α'], include_unchanged=True))