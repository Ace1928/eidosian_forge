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
def test_short_service_name(self):
    launchpad = NoNetworkLaunchpad.login_with('app name', 'staging')
    self.assertEqual(launchpad.passed_in_args['service_root'], 'https://api.staging.launchpad.net/')
    launchpad = NoNetworkLaunchpad.login_with('app name', uris.service_roots['staging'])
    self.assertEqual(launchpad.passed_in_args['service_root'], uris.service_roots['staging'])
    launchpad = ('app name', 'https://')
    self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, 'app name', 'foo')