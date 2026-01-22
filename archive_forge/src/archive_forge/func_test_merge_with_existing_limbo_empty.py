import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def test_merge_with_existing_limbo_empty(self):
    """Empty limbo dir is just cleaned up - see bug 427773"""
    wt = self.make_branch_and_tree('this')
    limbodir, deletiondir = self.get_limbodir_deletiondir(wt)
    os.mkdir(limbodir)
    self.do_merge(wt, wt)