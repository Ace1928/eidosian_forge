import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def lock_and_compare_all_current_dirstate(tree, lock_method):
    getattr(tree, lock_method)()
    state = tree.current_dirstate()
    self.assertFalse(state in known_dirstates)
    known_dirstates.add(state)
    tree.unlock()