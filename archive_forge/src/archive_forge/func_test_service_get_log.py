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
@ddt.data('*', 'cinder-api', 'cinder-volume', 'cinder-scheduler', 'cinder-backup')
@mock.patch('cinderclient.v3.services.ServiceManager.get_log_levels')
@mock.patch('cinderclient.shell_utils.print_list')
def test_service_get_log(self, binary, print_mock, get_levels_mock):
    server = 'host1'
    prefix = 'sqlalchemy'
    self.run_command('--os-volume-api-version 3.32 service-get-log --binary %s --server %s --prefix %s' % (binary, server, prefix))
    get_levels_mock.assert_called_once_with(binary, server, prefix)
    print_mock.assert_called_once_with(get_levels_mock.return_value, ('Binary', 'Host', 'Prefix', 'Level'))