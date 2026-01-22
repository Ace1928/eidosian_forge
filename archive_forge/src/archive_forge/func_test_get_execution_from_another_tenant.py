from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_get_execution_from_another_tenant(self):
    wf = self.workflow_create(self.wf_def)
    ex = self.execution_create(wf[0]['Name'])
    exec_id = self.get_field_value(ex, 'ID')
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'execution-get', params=exec_id)