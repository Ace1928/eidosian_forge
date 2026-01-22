import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_iter_changes_ignores_unversioned_dirs(self):
    """iter_changes should not descend into unversioned directories."""
    tree = self.make_branch_and_tree('.', format='dirstate')
    self.build_tree(['unversioned/', 'unversioned/a', 'unversioned/b/', 'versioned/', 'versioned/unversioned/', 'versioned/unversioned/a', 'versioned/unversioned/b/', 'versioned2/', 'versioned2/a', 'versioned2/unversioned/', 'versioned2/unversioned/a', 'versioned2/unversioned/b/'])
    tree.add(['versioned', 'versioned2', 'versioned2/a'])
    tree.commit('one', rev_id=b'rev-1')
    returned = []

    def walkdirs_spy(*args, **kwargs):
        for val in orig(*args, **kwargs):
            returned.append(val[0][0])
            yield val
    orig = self.overrideAttr(osutils, '_walkdirs_utf8', walkdirs_spy)
    basis = tree.basis_tree()
    tree.lock_read()
    self.addCleanup(tree.unlock)
    basis.lock_read()
    self.addCleanup(basis.unlock)
    changes = [c.path for c in tree.iter_changes(basis, want_unversioned=True)]
    self.assertEqual([(None, 'unversioned'), (None, 'versioned/unversioned'), (None, 'versioned2/unversioned')], changes)
    self.assertEqual([b'', b'versioned', b'versioned2'], returned)
    del returned[:]
    changes = [c[1] for c in tree.iter_changes(basis)]
    self.assertEqual([], changes)
    self.assertEqual([b'', b'versioned', b'versioned2'], returned)