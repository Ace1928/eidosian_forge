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
def test_merge_metadata_vs_deletion(self):
    """Conflict deletion vs metadata change"""
    a_wt = self.make_branch_and_tree('a')
    with open('a/file', 'wb') as f:
        f.write(b'contents\n')
    a_wt.add('file')
    a_wt.commit('r0')
    self.run_bzr('branch a b')
    b_wt = WorkingTree.open('b')
    os.chmod('b/file', 493)
    os.remove('a/file')
    a_wt.commit('removed a')
    self.assertEqual(a_wt.branch.revno(), 2)
    self.assertFalse(os.path.exists('a/file'))
    b_wt.commit('exec a')
    a_wt.merge_from_branch(b_wt.branch, b_wt.last_revision(), b'null:')
    self.assertTrue(os.path.exists('a/file'))