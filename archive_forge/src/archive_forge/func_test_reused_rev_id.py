import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_reused_rev_id(self):
    """Test that a revision id cannot be reused in a branch"""
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    wt.commit('initial', rev_id=b'test@rev-1', allow_pointless=True)
    self.assertRaises(Exception, wt.commit, message='reused id', rev_id=b'test@rev-1', allow_pointless=True)