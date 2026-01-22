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
@ddt.data({'cmd': '--os-volume-api-version 3.47 create --backup-id 1234', 'update': {'backup_id': '1234'}}, {'cmd': '--os-volume-api-version 3.47 create 2', 'update': {'size': 2}})
@ddt.unpack
def test_create_volume_with_backup(self, cmd, update):
    self.run_command(cmd)
    self.assert_called('GET', '/volumes/1234')
    expected = {'volume': {'imageRef': None, 'size': None, 'availability_zone': None, 'source_volid': None, 'consistencygroup_id': None, 'name': None, 'snapshot_id': None, 'metadata': {}, 'volume_type': None, 'description': None, 'backup_id': None}}
    expected['volume'].update(update)
    self.assert_called_anytime('POST', '/volumes', body=expected)