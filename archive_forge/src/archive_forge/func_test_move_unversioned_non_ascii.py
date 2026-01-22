import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_unversioned_non_ascii(self):
    """Check error when moving an unversioned non-ascii file"""
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['§', 'dir/'])
    tree.add('dir')
    e = self.assertRaises(errors.BzrMoveFailedError, tree.move, ['§'], 'dir')
    self.assertIsInstance(e.extra, errors.NotVersionedError)
    self.assertEqual(e.extra.path, '§')