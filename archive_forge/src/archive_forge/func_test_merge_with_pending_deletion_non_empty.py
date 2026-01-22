import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_merge_with_pending_deletion_non_empty(self):
    """Also see bug 427773"""
    wt = self.make_branch_and_tree('this')
    limbodir, deletiondir = self.get_limbodir_deletiondir(wt)
    os.mkdir(deletiondir)
    os.mkdir(os.path.join(deletiondir, 'something'))
    self.assertRaises(errors.ExistingPendingDeletion, self.do_merge, wt, wt)
    self.assertRaises(errors.LockError, wt.unlock)