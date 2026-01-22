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
@ddt.data({'volume': '1234', 'name': None, 'volume_type': None, 'availability_zone': None}, {'volume': '1234', 'name': 'ignored', 'volume_type': None, 'availability_zone': None}, {'volume': None, 'name': 'sample-volume', 'volume_type': 'sample-type', 'availability_zone': None}, {'volume': None, 'name': 'sample-volume', 'volume_type': None, 'availability_zone': 'az1'}, {'volume': None, 'name': 'sample-volume', 'volume_type': None, 'availability_zone': 'different-az'}, {'volume': None, 'name': None, 'volume_type': None, 'availability_zone': 'different-az'})
@ddt.unpack
@mock.patch('cinderclient.shell_utils.print_dict')
@mock.patch('cinderclient.tests.unit.v3.fakes_base._stub_restore')
def test_do_backup_restore(self, mock_stub_restore, mock_print_dict, volume, name, volume_type, availability_zone):
    cmd = '--os-volume-api-version 3.47 backup-restore 1234'
    if volume:
        cmd += ' --volume %s' % volume
    if name:
        cmd += ' --name %s' % name
    if volume_type:
        cmd += ' --volume-type %s' % volume_type
    if availability_zone:
        cmd += ' --availability-zone %s' % availability_zone
    if name or volume:
        volume_name = 'sample-volume'
    else:
        volume_name = 'restore_backup_1234'
    mock_stub_restore.return_value = {'volume_id': '1234', 'volume_name': volume_name}
    self.run_command(cmd)
    if volume_type or availability_zone == 'different-az':
        mock_stub_restore.assert_not_called()
    else:
        mock_stub_restore.assert_called_once()
    mock_print_dict.assert_called_once_with({'backup_id': '1234', 'volume_id': '1234', 'volume_name': volume_name})