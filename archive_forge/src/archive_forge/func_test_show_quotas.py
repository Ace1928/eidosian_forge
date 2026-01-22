from troveclient.osc.v1 import database_quota
from troveclient.tests.osc.v1 import fakes
def test_show_quotas(self):
    args = ['tenant_id']
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.values, data)