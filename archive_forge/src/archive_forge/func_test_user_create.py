from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_users
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_user_create(self, mock_find):
    args = ['instance1', 'user1', 'password1']
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, [])
    result = self.cmd.take_action(parsed_args)
    user = {'name': 'user1', 'password': 'password1', 'databases': []}
    self.user_client.create.assert_called_with('instance1', [user])
    self.assertIsNone(result)