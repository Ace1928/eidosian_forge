from troveclient import common
from troveclient.osc.v1 import database_limits
from troveclient.tests.osc.v1 import fakes
def test_limit_list_defaults(self):
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.limit_client.list.assert_called_once_with()
    self.assertEqual(self.columns, columns)
    self.assertEqual([self.non_absolute_values], data)