import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_get(self):
    execution = self.execution_create(self.direct_wf['Name'])
    exec_id = self.get_field_value(execution, 'ID')
    execution = self.mistral_admin('execution-get', params='{0}'.format(exec_id))
    gotten_id = self.get_field_value(execution, 'ID')
    wf_name = self.get_field_value(execution, 'Workflow name')
    wf_id = self.get_field_value(execution, 'Workflow ID')
    self.assertIsNotNone(wf_id)
    self.assertEqual(exec_id, gotten_id)
    self.assertEqual(self.direct_wf['Name'], wf_name)