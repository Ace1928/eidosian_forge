from unittest import mock
import testtools
from heatclient.common import hook_utils
import heatclient.v1.shell as shell
def test_create_hooks_in_args(self):
    type(self.args).pre_create = mock.PropertyMock(return_value=['bp', 'another_bp'])
    shell.do_stack_create(self.client, self.args)
    self.assertEqual(1, self.client.stacks.create.call_count)
    expected_hooks = {'bp': {'hooks': 'pre-create'}, 'another_bp': {'hooks': 'pre-create'}}
    actual_hooks = self.client.stacks.create.call_args[1]['environment']['resource_registry']['resources']
    self.assertEqual(expected_hooks, actual_hooks)