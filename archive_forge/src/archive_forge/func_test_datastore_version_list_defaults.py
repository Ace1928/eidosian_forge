from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
def test_datastore_version_list_defaults(self):
    args = ['mysql']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.datastore_version_client.list.assert_called_once_with(args[0])
    self.assertEqual(self.columns, columns)
    self.assertEqual([self.values], data)