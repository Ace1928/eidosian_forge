import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def tree_iter_changes(tree, files):
    return list(tree.iter_changes(tree.basis_tree(), specific_files=files, require_versioned=True))