import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_act_execution_create_delete(self):
    action_ex = self.mistral_admin('run-action', params="std.echo '{0}' --save-result".format('{"output": "Hello!"}'))
    action_ex_id = self.get_field_value(action_ex, 'ID')
    self.assertTableStruct(action_ex, ['Field', 'Value'])
    name = self.get_field_value(action_ex, 'Name')
    wf_name = self.get_field_value(action_ex, 'Workflow name')
    task_name = self.get_field_value(action_ex, 'Task name')
    self.assertEqual('std.echo', name)
    self.assertEqual('None', wf_name)
    self.assertEqual('None', task_name)
    action_exs = self.mistral_admin('action-execution-list')
    self.assertIn(action_ex_id, [ex['ID'] for ex in action_exs])
    self.mistral_admin('action-execution-delete', params=action_ex_id)
    action_exs = self.mistral_admin('action-execution-list')
    self.assertNotIn(action_ex_id, [ex['ID'] for ex in action_exs])