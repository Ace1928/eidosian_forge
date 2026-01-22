import os
from breezy import errors, osutils, tests
from breezy import transport as _mod_transport
from breezy.tests import features
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_rename_over_already_versioned_non_ascii(self):
    """Check error renaming over an already versioned non-ascii file"""
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', '§'])
    tree.add(['a', '§'])
    e = self.assertRaises(errors.BzrMoveFailedError, tree.rename_one, 'a', '§')
    self.assertIsInstance(e.extra, errors.AlreadyVersionedError)
    self.assertEqual(e.extra.path, '§')