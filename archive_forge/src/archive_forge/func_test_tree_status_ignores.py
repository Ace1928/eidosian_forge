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
def test_tree_status_ignores(self):
    """Tests branch status with ignores"""
    wt = self.make_branch_and_tree('.')
    self.run_bzr('ignore *~')
    wt.commit('commit .bzrignore')
    self.build_tree(['foo.c', 'foo.c~'])
    self.assertStatus(['unknown:\n', '  foo.c\n'], wt)
    self.assertStatus(['?   foo.c\n'], wt, short=True)