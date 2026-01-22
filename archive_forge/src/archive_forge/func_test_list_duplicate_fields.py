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
@mock.patch('cinderclient.shell_utils.print_list')
def test_list_duplicate_fields(self, mock_print):
    self.run_command('list --field Status,id,Size,status')
    self.assert_called('GET', '/volumes/detail')
    key_list = ['ID', 'Status', 'Size']
    mock_print.assert_called_once_with(mock.ANY, key_list, exclude_unavailable=True, sortby_index=0)