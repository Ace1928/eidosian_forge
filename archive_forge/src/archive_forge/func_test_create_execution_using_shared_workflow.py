from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_create_execution_using_shared_workflow(self):
    self._update_shared_workflow(new_status='accepted')
    execution = self.execution_create(self.wf[0]['ID'], admin=False)
    wf_name = self.get_field_value(execution, 'Workflow name')
    self.assertEqual(self.wf[0]['Name'], wf_name)