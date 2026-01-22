import os
import re
import sys
import breezy
from breezy import osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.tests import TestCaseWithTransport
from breezy.tests.http_utils import TestCaseWithWebserver
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_external_command(self):
    """Test that external commands can be run by setting the path
        """
    cmd_name = 'test-command'
    if sys.platform == 'win32':
        cmd_name += '.bat'
    self.overrideEnv('BZRPATH', None)
    f = open(cmd_name, 'wb')
    if sys.platform == 'win32':
        f.write(b'@echo off\n')
    else:
        f.write(b'#!/bin/sh\n')
    f.close()
    os.chmod(cmd_name, 493)
    self.run_bzr(cmd_name, retcode=3)
    self.overrideEnv('BZRPATH', '.')
    self.run_bzr(cmd_name)
    self.overrideEnv('BZRPATH', os.pathsep)
    self.run_bzr(cmd_name, retcode=3)