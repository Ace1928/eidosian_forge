from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
@mock.patch('troveclient.utils.get_resource_id_by_name')
def test_incremental_backup_create(self, mock_find):
    args = ['bk-1234-2', '--instance', '1234', '--description', 'backup 1234', '--parent', '1234-1', '--incremental']
    mock_find.return_value = 'fake-instance-id'
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.backup_client.create.assert_called_with('bk-1234-2', 'fake-instance-id', description='backup 1234', parent_id='1234-1', incremental=True, swift_container=None)