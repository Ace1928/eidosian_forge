from contextlib import contextmanager
import os
import shutil
import socket
import stat
import tempfile
import unittest
import warnings
from lazr.restfulclient.resource import ServiceRoot
from launchpadlib.credentials import (
from launchpadlib import uris
import launchpadlib.launchpad
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
def test_dirs_created_are_secure(self):
    launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
    NoNetworkLaunchpad.login_with('not important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
    self.assertTrue(os.path.isdir(launchpadlib_dir))
    statinfo = os.stat(launchpadlib_dir)
    mode = stat.S_IMODE(statinfo.st_mode)
    self.assertEqual(mode, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)