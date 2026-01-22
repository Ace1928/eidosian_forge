from unittest import mock
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import test_utils
from cinderclient.tests.unit import utils
from cinderclient.v3 import client
def test_get_highest_version(self):
    self._mock_returned_server_version('3.14', '3.0')
    highest_version = api_versions.get_highest_version(self.fake_client)
    self.assertEqual('3.14', highest_version.get_string())
    self.assertTrue(self.fake_client.services.server_api_version.called)