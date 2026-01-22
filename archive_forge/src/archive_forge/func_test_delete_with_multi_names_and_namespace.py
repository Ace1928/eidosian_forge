from unittest import mock
from mistralclient.api.v2 import actions
from mistralclient.commands.v2 import actions as action_cmd
from mistralclient.commands.v2 import base as cmd_base
from mistralclient.tests.unit import base
def test_delete_with_multi_names_and_namespace(self):
    self.call(action_cmd.Delete, app_args=['name1', 'name2', '--namespace', 'test_namespace'])
    self.assertEqual(2, self.client.actions.delete.call_count)
    self.assertEqual([mock.call('name1', namespace='test_namespace'), mock.call('name2', namespace='test_namespace')], self.client.actions.delete.call_args_list)