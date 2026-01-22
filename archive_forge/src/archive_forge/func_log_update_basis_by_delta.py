import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def log_update_basis_by_delta(delta, new_revid):
    called.append(new_revid)
    return orig_update(delta, new_revid)