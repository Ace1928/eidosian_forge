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
@ddt.data({'cmd': '1234 1233', 'body': {'instance_uuid': '1233', 'connector': {}, 'volume_uuid': '1234', 'mode': 'ro'}}, {'cmd': '1234 1233 --connect True --ip 10.23.12.23 --host server01 --platform x86_xx --ostype 123 --multipath true --mountpoint /123 --initiator aabbccdd', 'body': {'instance_uuid': '1233', 'connector': {'ip': '10.23.12.23', 'host': 'server01', 'os_type': '123', 'multipath': 'true', 'mountpoint': '/123', 'initiator': 'aabbccdd', 'platform': 'x86_xx'}, 'volume_uuid': '1234', 'mode': 'ro'}}, {'cmd': 'abc 1233', 'body': {'instance_uuid': '1233', 'connector': {}, 'volume_uuid': '1234', 'mode': 'ro'}}, {'cmd': '1234', 'body': {'connector': {}, 'volume_uuid': '1234', 'mode': 'ro'}}, {'cmd': '1234 --connect True --ip 10.23.12.23 --host server01 --platform x86_xx --ostype 123 --multipath true --mountpoint /123 --initiator aabbccdd', 'body': {'connector': {'ip': '10.23.12.23', 'host': 'server01', 'os_type': '123', 'multipath': 'true', 'mountpoint': '/123', 'initiator': 'aabbccdd', 'platform': 'x86_xx'}, 'volume_uuid': '1234', 'mode': 'ro'}})
@mock.patch('cinderclient.utils.find_resource')
@ddt.unpack
def test_attachment_create_with_mode(self, mock_find_volume, cmd, body):
    mock_find_volume.return_value = volumes.Volume(self, {'id': '1234'}, loaded=True)
    command = '--os-volume-api-version 3.54 attachment-create --mode ro '
    command += cmd
    self.run_command(command)
    expected = {'attachment': body}
    self.assertTrue(mock_find_volume.called)
    self.assert_called('POST', '/attachments', body=expected)