import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
def test_status_out_of_date(self):
    """Simulate status of out-of-date tree after remote push"""
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', b'foo\n')])
    with tree.lock_write():
        tree.add(['a'])
        tree.commit('add test file')
        tree.set_last_revision(b'0')
    out, err = self.run_bzr('status')
    self.assertEqual("working tree is out of date, run 'brz update'\n", err)