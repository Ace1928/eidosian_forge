import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_executions_list_with_rootsonly(self):
    wrapping_wf = self.workflow_create(self.wf_wrapping_wf)
    wrapping_wf_ex = self.execution_create(wrapping_wf[-1]['Name'])
    wrapping_wf_ex_id = self.get_field_value(wrapping_wf_ex, 'ID')
    wf_execs = self.mistral_cli(True, 'execution-list', params='--rootsonly')
    self.assertEqual(1, len(wf_execs))
    wf_exec = wf_execs[0]
    self.assertEqual(wrapping_wf_ex_id, wf_exec['ID'])