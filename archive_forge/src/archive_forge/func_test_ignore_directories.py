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
def test_ignore_directories(self):
    """ignoring a directory should ignore directory tree.

        Also check that trailing slashes on directories are stripped.
        """
    self.run_bzr('init')
    self.build_tree(['dir1/', 'dir1/foo', 'dir2/', 'dir2/bar', 'dir3/', 'dir3/baz'])
    self.run_bzr(['ignore', 'dir1', 'dir2/', 'dir4\\'])
    self.check_file_contents('.bzrignore', b'dir1\ndir2\ndir4\n')
    self.assertEqual(self.run_bzr('unknowns')[0], 'dir3\n')