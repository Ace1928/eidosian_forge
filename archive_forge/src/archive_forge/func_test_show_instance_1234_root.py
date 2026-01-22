from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient.osc.v1 import database_root
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_show_instance_1234_root(self, mock_find):
    self.root_client.is_instance_root_enabled.return_value = self.data['instance']
    args = ['1234']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(('True',), data)