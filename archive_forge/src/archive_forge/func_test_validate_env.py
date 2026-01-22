from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import template
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import template_versions
def test_validate_env(self):
    arglist = ['-t', self.template_path, '-e', self.env_path]
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(1, self.stack_client.validate.call_count)
    args = self.stack_client.validate.call_args[1]
    self.assertEqual(args.get('environment'), {'parameters': {}})
    self.assertIn(self.env_path, args.get('environment_files')[0])
    self.assertEqual([], columns)
    self.assertEqual([], data)