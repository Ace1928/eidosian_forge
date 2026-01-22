import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_task_get(self):
    wf_ex = self.execution_create(self.direct_wf['Name'])
    wf_ex_id = self.get_field_value(wf_ex, 'ID')
    tasks = self.mistral_admin('task-list', params=wf_ex_id)
    created_task_id = tasks[-1]['ID']
    fetched_task = self.mistral_admin('task-get', params=created_task_id)
    fetched_task_id = self.get_field_value(fetched_task, 'ID')
    fetched_task_wf_namespace = self.get_field_value(fetched_task, 'Workflow namespace')
    task_execution_id = self.get_field_value(fetched_task, 'Workflow Execution ID')
    self.assertEqual(created_task_id, fetched_task_id)
    self.assertEqual('', fetched_task_wf_namespace)
    self.assertEqual(wf_ex_id, task_execution_id)