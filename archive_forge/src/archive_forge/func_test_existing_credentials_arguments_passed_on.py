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
def test_existing_credentials_arguments_passed_on(self):
    os.makedirs(os.path.join(self.temp_dir, 'api.example.com', 'credentials'))
    credentials_file_path = os.path.join(self.temp_dir, 'api.example.com', 'credentials', 'app name')
    credentials = Credentials('app name', consumer_secret='consumer_secret:42', access_token=AccessToken('access_key:84', 'access_secret:168'))
    credentials.save_to_path(credentials_file_path)
    timeout = object()
    proxy_info = object()
    version = 'foo'
    launchpad = NoNetworkLaunchpad.login_with('app name', launchpadlib_dir=self.temp_dir, service_root=SERVICE_ROOT, timeout=timeout, proxy_info=proxy_info, version=version)
    expected_arguments = dict(service_root=SERVICE_ROOT, timeout=timeout, proxy_info=proxy_info, version=version, cache=os.path.join(self.temp_dir, 'api.example.com', 'cache'))
    for key, expected in expected_arguments.items():
        actual = launchpad.passed_in_args[key]
        self.assertEqual(actual, expected)