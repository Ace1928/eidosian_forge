import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_iter_changes_unversioned_non_ascii(self):
    """Unversioned non-ascii paths should be reported as unicode"""
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('f', b'')])
    tree.add(['f'], ids=[b'f-id'])

    def tree_iter_changes(tree, files):
        return list(tree.iter_changes(tree.basis_tree(), specific_files=files, require_versioned=True))
    tree.lock_read()
    self.addCleanup(tree.unlock)
    e = self.assertRaises(errors.PathsNotVersionedError, tree_iter_changes, tree, ['§', 'π'])
    self.assertEqual(set(e.paths), {'§', 'π'})