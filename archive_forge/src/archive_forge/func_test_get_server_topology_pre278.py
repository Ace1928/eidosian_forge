import base64
import io
import os
import tempfile
from unittest import mock
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import floatingips
from novaclient.tests.unit.fixture_data import servers as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
def test_get_server_topology_pre278(self):
    self.cs.api_version = api_versions.APIVersion('2.77')
    s = self.cs.servers.get(1234)
    self.assertRaises(exceptions.VersionNotFoundForAPIMethod, s.topology)