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
def test_commit_move(self):
    """Test commit of revisions with moved files and directories"""
    eq = self.assertEqual
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    r1 = b'test@rev-1'
    self.build_tree(['hello', 'a/', 'b/'])
    wt.add(['hello', 'a', 'b'], ids=[b'hello-id', b'a-id', b'b-id'])
    wt.commit('initial', rev_id=r1, allow_pointless=False)
    wt.move(['hello'], 'a')
    r2 = b'test@rev-2'
    wt.commit('two', rev_id=r2, allow_pointless=False)
    wt.lock_read()
    try:
        self.check_tree_shape(wt, ['a/', 'a/hello', 'b/'])
    finally:
        wt.unlock()
    wt.move(['b'], 'a')
    r3 = b'test@rev-3'
    wt.commit('three', rev_id=r3, allow_pointless=False)
    wt.lock_read()
    try:
        self.check_tree_shape(wt, ['a/', 'a/hello', 'a/b/'])
        self.check_tree_shape(b.repository.revision_tree(r3), ['a/', 'a/hello', 'a/b/'])
    finally:
        wt.unlock()
    wt.move(['a/hello'], 'a/b')
    r4 = b'test@rev-4'
    wt.commit('four', rev_id=r4, allow_pointless=False)
    wt.lock_read()
    try:
        self.check_tree_shape(wt, ['a/', 'a/b/hello', 'a/b/'])
    finally:
        wt.unlock()
    inv = b.repository.get_inventory(r4)
    eq(inv.get_entry(b'hello-id').revision, r4)
    eq(inv.get_entry(b'a-id').revision, r1)
    eq(inv.get_entry(b'b-id').revision, r3)