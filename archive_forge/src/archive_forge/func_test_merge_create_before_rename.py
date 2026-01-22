import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
def test_merge_create_before_rename(self):
    """create before rename, target parents before children

        This case requires that you must not do move-into-place
        before creates, and that you must not do children after
        parents:

        $ touch foo
        $ bzr add foo
        $ bzr commit
        $ bzr mkdir bar
        $ bzr add bar
        $ bzr mv foo bar/foo
        $ bzr commit
        """
    os.mkdir('a')
    a_wt = self.make_branch_and_tree('a')
    with open('a/foo', 'wb') as f:
        f.write(b'A/FOO')
    a_wt.add('foo')
    a_wt.commit('added foo')
    self.run_bzr('branch a b')
    b_wt = WorkingTree.open('b')
    os.mkdir('b/bar')
    b_wt.add('bar')
    b_wt.rename_one('foo', 'bar/foo')
    b_wt.commit('created bar dir, moved foo into bar')
    a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))