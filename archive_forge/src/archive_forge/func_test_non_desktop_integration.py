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
def test_non_desktop_integration(self):
    launchpad = NoNetworkLaunchpad.login_with(consumer_name='consumer', allow_access_levels=['FOO'])
    self.assertEqual(launchpad.credentials.consumer.key, 'consumer')
    self.assertEqual(launchpad.credentials.consumer.application_name, None)
    self.assertEqual(launchpad.authorization_engine.allow_access_levels, ['FOO'])