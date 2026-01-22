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
@mock.patch('cinderclient.shell_utils._poll_for_status')
def test_create_with_poll(self, poll_method):
    self.run_command('create --poll 1')
    self.assert_called_anytime('GET', '/volumes/1234')
    volume = self.shell.cs.volumes.get('1234')
    info = dict()
    info.update(volume._info)
    self.assertEqual(1, poll_method.call_count)
    timeout_period = 3600
    poll_method.assert_has_calls([mock.call(self.shell.cs.volumes.get, 1234, info, 'creating', ['available'], timeout_period, self.shell.cs.client.global_request_id, self.shell.cs.messages)])