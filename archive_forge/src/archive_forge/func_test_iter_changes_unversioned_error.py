import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_iter_changes_unversioned_error(self):
    """ Check if a PathsNotVersionedError is correctly raised and the
            paths list contains all unversioned entries only.
        """
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/bar', b'')])
    tree.add(['bar'], ids=[b'bar-id'])
    tree.lock_read()
    self.addCleanup(tree.unlock)

    def tree_iter_changes(files):
        return [c for c in tree.iter_changes(tree.basis_tree(), specific_files=files, require_versioned=True)]
    e = self.assertRaises(errors.PathsNotVersionedError, tree_iter_changes, ['bar', 'foo'])
    self.assertEqual(e.paths, ['foo'])