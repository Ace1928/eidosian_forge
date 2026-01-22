import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_task_list_with_limit(self):
    wf_exec = self.execution_create('%s input task_name' % self.reverse_wf['Name'])
    exec_id = self.get_field_value(wf_exec, 'ID')
    self.assertTrue(self.wait_execution_success(exec_id))
    tasks = self.parser.listing(self.mistral('task-list'))
    tasks = self.parser.listing(self.mistral('task-list', params='--limit 1'))
    self.assertEqual(1, len(tasks))