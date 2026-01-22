import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_ex_update_both_state_and_description(self):
    wf = self.workflow_create(self.wf_def)
    execution = self.execution_create(params=wf[0]['Name'])
    exec_id = self.get_field_value(execution, 'ID')
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-update', params='%s -s ERROR -d update' % exec_id)