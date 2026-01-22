import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_create_with_input_and_start_task(self):
    execution = self.execution_create('%s input task_name' % self.reverse_wf['Name'])
    exec_id = self.get_field_value(execution, 'ID')
    result = self.wait_execution_success(exec_id)
    self.assertTrue(result)