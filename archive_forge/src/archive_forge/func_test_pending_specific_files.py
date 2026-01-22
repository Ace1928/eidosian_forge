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
def test_pending_specific_files(self):
    """With a specific file list, pending merges are not shown."""
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/a', b'content of a\n')])
    tree.add('a')
    r1_id = tree.commit('one')
    alt = tree.controldir.sprout('alt').open_workingtree()
    self.build_tree_contents([('alt/a', b'content of a\nfrom alt\n')])
    alt_id = alt.commit('alt')
    tree.merge_from_branch(alt.branch)
    output = self.make_utf8_encoded_stringio()
    show_tree_status(tree, to_file=output)
    self.assertContainsRe(output.getvalue(), b'pending merge')
    out, err = self.run_bzr('status tree/a')
    self.assertNotContainsRe(out, 'pending merge')