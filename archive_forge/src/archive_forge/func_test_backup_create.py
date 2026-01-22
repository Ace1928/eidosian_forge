from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
@mock.patch('troveclient.utils.get_resource_id_by_name')
def test_backup_create(self, mock_find):
    args = ['bk-1234-1', '--instance', '1234']
    mock_find.return_value = 'fake-instance-id'
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.backup_client.create.assert_called_with('bk-1234-1', 'fake-instance-id', description=None, parent_id=None, incremental=False, swift_container=None)