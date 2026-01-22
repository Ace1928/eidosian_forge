from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
def test_datastore_list_defaults(self):
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.datastore_client.list.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual([self.values], data)