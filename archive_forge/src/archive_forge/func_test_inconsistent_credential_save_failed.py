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
def test_inconsistent_credential_save_failed(self):

    def callback1():
        pass
    store = KeyringCredentialStore(credential_save_failed=callback1)

    def callback2():
        pass
    self.assertRaises(ValueError, NoNetworkLaunchpad.login_with, 'app name', credential_store=store, credential_save_failed=callback2)