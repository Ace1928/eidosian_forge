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
def test_merge_swapping_renames(self):
    a_wt = self.make_branch_and_tree('a')
    with open('a/un', 'wb') as f:
        f.write(b'UN')
    with open('a/deux', 'wb') as f:
        f.write(b'DEUX')
    a_wt.add('un')
    a_wt.add('deux')
    a_wt.commit('r0', rev_id=b'r0')
    self.run_bzr('branch a b')
    b_wt = WorkingTree.open('b')
    b_wt.rename_one('un', 'tmp')
    b_wt.rename_one('deux', 'un')
    b_wt.rename_one('tmp', 'deux')
    b_wt.commit('r1', rev_id=b'r1')
    self.assertEqual(0, len(a_wt.merge_from_branch(b_wt.branch, b_wt.branch.last_revision(), b_wt.branch.get_rev_id(1))))
    self.assertPathExists('a/un')
    self.assertTrue('a/deux')
    self.assertFalse(os.path.exists('a/tmp'))
    with open('a/un') as f:
        self.assertEqual(f.read(), 'DEUX')
    with open('a/deux') as f:
        self.assertEqual(f.read(), 'UN')