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
def test_snapshot_list(self, mock_print_list):
    """Ensure we always present all existing fields when listing snaps."""
    self.run_command('--os-volume-api-version 3.65 snapshot-list')
    self.assert_called('GET', '/snapshots/detail')
    columns = ['ID', 'Volume ID', 'Status', 'Name', 'Size', 'Consumes Quota', 'User ID']
    mock_print_list.assert_called_once_with(mock.ANY, columns, exclude_unavailable=True, sortby_index=0)