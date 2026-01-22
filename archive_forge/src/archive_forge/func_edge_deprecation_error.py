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
@contextmanager
def edge_deprecation_error(self):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield
        self.assertEqual(len(caught), 1)
        warning, = caught
        self.assertTrue(issubclass(warning.category, DeprecationWarning))
        self.assertIn('no longer exists', str(warning))