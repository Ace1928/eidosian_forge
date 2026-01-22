from troveclient.osc.v1 import database_flavors
from troveclient.tests.osc.v1 import fakes
def test_flavor_show_defaults(self):
    args = ['m1.tiny']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.values, data)