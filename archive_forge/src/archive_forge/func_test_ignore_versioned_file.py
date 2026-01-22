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
def test_ignore_versioned_file(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b'])
    tree.add('a')
    out, err = self.run_bzr('ignore a')
    self.assertEqual(out, "Warning: the following files are version controlled and match your ignore pattern:\na\nThese files will continue to be version controlled unless you 'brz remove' them.\n")
    out, err = self.run_bzr('ignore b')
    self.assertEqual(out, '')
    tree.add('b')
    out, err = self.run_bzr('ignore *')
    self.assertEqual(out, "Warning: the following files are version controlled and match your ignore pattern:\n.bzrignore\na\nb\nThese files will continue to be version controlled unless you 'brz remove' them.\n")