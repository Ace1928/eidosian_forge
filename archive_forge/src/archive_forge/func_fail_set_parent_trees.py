import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def fail_set_parent_trees(trees, ghosts):
    raise AssertionError('dirstate.set_parent_trees() was called')