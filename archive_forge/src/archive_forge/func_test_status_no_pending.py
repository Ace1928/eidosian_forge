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
def test_status_no_pending(self):
    a_tree = self.make_branch_and_tree('a')
    self.build_tree(['a/a'])
    a_tree.add('a')
    a_tree.commit('a')
    b_tree = a_tree.controldir.sprout('b').open_workingtree()
    self.build_tree(['b/b'])
    b_tree.add('b')
    b_tree.commit('b')
    self.run_bzr('merge ../b', working_dir='a')
    out, err = self.run_bzr('status --no-pending', working_dir='a')
    self.assertEqual(out, 'added:\n  b\n')