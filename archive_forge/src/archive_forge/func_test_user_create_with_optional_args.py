from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_user_create_with_optional_args(self, mock_find):
    args = ['instance2', 'user2', 'password2', '--host', '1.1.1.1', '--databases', 'db1', 'db2']
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, [])
    result = self.cmd.take_action(parsed_args)
    user = {'name': 'user2', 'password': 'password2', 'host': '1.1.1.1', 'databases': [{'name': 'db1'}, {'name': 'db2'}]}
    self.user_client.create.assert_called_with('instance2', [user])
    self.assertIsNone(result)