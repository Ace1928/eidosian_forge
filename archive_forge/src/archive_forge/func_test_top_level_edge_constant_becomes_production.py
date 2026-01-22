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
def test_top_level_edge_constant_becomes_production(self):
    with self.edge_deprecation_error():
        self.assertEqual(uris.lookup_service_root(uris.EDGE_SERVICE_ROOT), uris.lookup_service_root('production'))