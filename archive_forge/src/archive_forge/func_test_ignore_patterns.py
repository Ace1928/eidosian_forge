import os
import re
import breezy
from breezy import ignores, osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.osutils import pathjoin
from breezy.tests import TestCaseWithTransport
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_ignore_patterns(self):
    tree = self.make_branch_and_tree('.')
    self.assertEqual(list(tree.unknowns()), [])
    ignores._set_user_ignores(['*.tmp'])
    self.build_tree_contents([('foo.tmp', b'.tmp files are ignored by default')])
    self.assertEqual(list(tree.unknowns()), [])
    self.build_tree_contents([('foo.c', b'int main() {}')])
    self.assertEqual(list(tree.unknowns()), ['foo.c'])
    tree.add('foo.c')
    self.assertEqual(list(tree.unknowns()), [])
    self.build_tree_contents([('foo.blah', b'blah')])
    self.assertEqual(list(tree.unknowns()), ['foo.blah'])
    self.run_bzr('ignore *.blah')
    self.assertEqual(list(tree.unknowns()), [])
    self.check_file_contents('.bzrignore', b'*.blah\n')
    self.build_tree_contents([('garh', b'garh')])
    self.assertEqual(list(tree.unknowns()), ['garh'])
    self.run_bzr('ignore garh')
    self.assertEqual(list(tree.unknowns()), [])
    self.check_file_contents('.bzrignore', b'*.blah\ngarh\n')