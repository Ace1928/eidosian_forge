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
def test_merge_delete_and_add_same(self):
    a_wt = self.make_branch_and_tree('a')
    with open('a/file', 'wb') as f:
        f.write(b'THIS')
    a_wt.add('file')
    a_wt.commit('r0')
    self.run_bzr('branch a b')
    b_wt = WorkingTree.open('b')
    os.remove('b/file')
    b_wt.commit('r1')
    with open('b/file', 'wb') as f:
        f.write(b'THAT')
    b_wt.add('file')
    b_wt.commit('r2')
    a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))
    self.assertTrue(os.path.exists('a/file'))
    with open('a/file') as f:
        self.assertEqual(f.read(), 'THAT')