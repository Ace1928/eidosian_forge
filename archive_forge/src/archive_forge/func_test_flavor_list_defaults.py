from troveclient.osc.v1 import database_flavors
from troveclient.tests.osc.v1 import fakes
def test_flavor_list_defaults(self):
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, values = self.cmd.take_action(parsed_args)
    self.flavor_client.list.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual([self.values], values)