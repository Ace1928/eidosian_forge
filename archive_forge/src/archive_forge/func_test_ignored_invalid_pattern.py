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
def test_ignored_invalid_pattern(self):
    """Ensure graceful handling for invalid ignore pattern.

        Test case for #300062.
        Invalid pattern should show clear error message.
        Invalid pattern should not be added to .bzrignore file.
        """
    tree = self.make_branch_and_tree('.')
    out, err = self.run_bzr(['ignore', 'RE:*.cpp', 'foo', 'RE:['], 3)
    self.assertEqual(out, '')
    self.assertContainsRe(err, 'Invalid ignore pattern.*RE:\\*\\.cpp.*RE:\\[', re.DOTALL)
    self.assertNotContainsRe(err, 'foo', re.DOTALL)
    self.assertFalse(os.path.isfile('.bzrignore'))