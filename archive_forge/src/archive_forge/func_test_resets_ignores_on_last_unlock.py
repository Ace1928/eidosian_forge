import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_resets_ignores_on_last_unlock(self):
    tree = self.make_workingtree()
    with tree.lock_read():
        with tree.lock_read():
            tree.is_ignored('foo')
        self.assertIsNot(None, tree._ignoreglobster)
    self.assertIs(None, tree._ignoreglobster)