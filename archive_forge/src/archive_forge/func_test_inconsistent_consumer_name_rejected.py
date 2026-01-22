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
def test_inconsistent_consumer_name_rejected(self):
    """Catch an attempt to specify inconsistent application_names."""
    engine = NoNetworkAuthorizationEngine(SERVICE_ROOT, None, consumer_name='consumer_name1')
    self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, 'consumer_name2', authorization_engine=engine)