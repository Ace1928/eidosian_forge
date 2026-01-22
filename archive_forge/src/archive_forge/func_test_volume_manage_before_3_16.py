from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_volume_manage_before_3_16(self):
    """Cluster optional argument was not acceptable."""
    self.assertRaises(exceptions.UnsupportedAttribute, self.run_command, 'manage host1 some_fake_name --cluster clustername--name foo --description bar --bootable --volume-type baz --availability-zone az --metadata k1=v1 k2=v2')