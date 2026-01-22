import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_without_username(self):
    """Ensure init works if username is not set.
        """
    self.overrideEnv('EMAIL', None)
    self.overrideEnv('BRZ_EMAIL', None)
    out, err = self.run_bzr(['init', 'foo'])
    self.assertEqual(err, '')
    self.assertTrue(os.path.exists('foo'))