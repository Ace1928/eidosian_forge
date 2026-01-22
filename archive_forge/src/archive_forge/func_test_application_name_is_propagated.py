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
def test_application_name_is_propagated(self):
    launchpadlib_dir = os.path.join(self.temp_dir, 'launchpadlib')
    launchpad = NoNetworkLaunchpad.login_with('very important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
    self.assertEqual(launchpad.credentials.consumer.application_name, 'very important')
    launchpad = NoNetworkLaunchpad.login_with('very important', service_root=SERVICE_ROOT, launchpadlib_dir=launchpadlib_dir)
    self.assertEqual(launchpad.credentials.consumer.application_name, 'very important')