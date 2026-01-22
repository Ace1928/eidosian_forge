import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_merge_with_pending_deletion_empty(self):
    wt = self.make_branch_and_tree('this')
    limbodir, deletiondir = self.get_limbodir_deletiondir(wt)
    os.mkdir(deletiondir)
    self.do_merge(wt, wt)