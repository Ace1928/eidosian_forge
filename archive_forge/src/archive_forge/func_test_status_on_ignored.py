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
def test_status_on_ignored(self):
    """Tests branch status on an unversioned file which is considered ignored.

        See https://bugs.launchpad.net/bzr/+bug/40103
        """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['test1.c', 'test1.c~', 'test2.c~'])
    result = self.run_bzr('status')[0]
    self.assertContainsRe(result, 'unknown:\n  test1.c\n')
    short_result = self.run_bzr('status --short')[0]
    self.assertContainsRe(short_result, '\\?   test1.c\n')
    result = self.run_bzr('status test1.c')[0]
    self.assertContainsRe(result, 'unknown:\n  test1.c\n')
    short_result = self.run_bzr('status --short test1.c')[0]
    self.assertContainsRe(short_result, '\\?   test1.c\n')
    result = self.run_bzr('status test1.c~')[0]
    self.assertContainsRe(result, 'ignored:\n  test1.c~\n')
    short_result = self.run_bzr('status --short test1.c~')[0]
    self.assertContainsRe(short_result, 'I   test1.c~\n')
    result = self.run_bzr('status test1.c~ test2.c~')[0]
    self.assertContainsRe(result, 'ignored:\n  test1.c~\n  test2.c~\n')
    short_result = self.run_bzr('status --short test1.c~ test2.c~')[0]
    self.assertContainsRe(short_result, 'I   test1.c~\nI   test2.c~\n')
    result = self.run_bzr('status test1.c test1.c~ test2.c~')[0]
    self.assertContainsRe(result, 'unknown:\n  test1.c\nignored:\n  test1.c~\n  test2.c~\n')
    short_result = self.run_bzr('status --short test1.c test1.c~ test2.c~')[0]
    self.assertContainsRe(short_result, '\\?   test1.c\nI   test1.c~\nI   test2.c~\n')