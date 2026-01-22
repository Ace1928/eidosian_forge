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
def test_merge_unrelated_conflicting(self):
    """Sucessfully merges unrelated branches with common names"""
    wta = self.make_branch_and_tree('a')
    a = wta.branch
    with open('a/file', 'wb') as f:
        f.write(b'contents\n')
    wta.add('file')
    wta.commit('a_revision', allow_pointless=False)
    wtb = self.make_branch_and_tree('b')
    b = wtb.branch
    with open('b/file', 'wb') as f:
        f.write(b'contents\n')
    wtb.add('file')
    b_rev = wtb.commit('b_revision', allow_pointless=False)
    wta.merge_from_branch(wtb.branch, b_rev, b'null:')
    self.assertTrue(os.path.lexists('a/file'))
    self.assertTrue(os.path.lexists('a/file.moved'))
    self.assertEqual([b_rev], wta.get_parent_ids()[1:])