import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_unversioned_non_ascii(self):
    """Check error when renaming an unversioned non-ascii file"""
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['§'])
    e = self.assertRaises(errors.BzrRenameFailedError, tree.rename_one, '§', 'b')
    self.assertIsInstance(e.extra, errors.NotVersionedError)
    self.assertEqual(e.extra.path, '§')