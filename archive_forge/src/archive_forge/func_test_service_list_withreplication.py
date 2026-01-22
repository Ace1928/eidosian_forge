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
@ddt.data(('3.0', None), ('3.6', None), ('3.7', True), ('3.7', False), ('3.7', ''))
@ddt.unpack
def test_service_list_withreplication(self, version, replication):
    command = '--os-volume-api-version %s service-list' % version
    if replication is not None:
        command += ' --withreplication %s' % replication
    self.run_command(command)
    self.assert_called('GET', '/os-services')