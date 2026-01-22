import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def lock_and_call_current_dirstate(tree, lock_method):
    getattr(tree, lock_method)()
    tree.current_dirstate()
    tree.unlock()