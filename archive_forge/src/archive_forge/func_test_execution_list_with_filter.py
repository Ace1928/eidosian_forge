import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_list_with_filter(self):
    wf_ex1 = self.execution_create(params='{0} -d "a"'.format(self.direct_wf['Name']))
    wf_ex1_id = self.get_field_value(wf_ex1, 'ID')
    self.execution_create(params='{0} -d "b"'.format(self.direct_wf['Name']))
    wf_execs = self.mistral_cli(True, 'execution-list')
    self.assertTableStruct(wf_execs, ['ID', 'Workflow name', 'Workflow ID', 'State', 'Created at', 'Updated at'])
    self.assertEqual(2, len(wf_execs))
    wf_execs = self.mistral_cli(True, 'execution-list', params='--filter description=a')
    self.assertTableStruct(wf_execs, ['ID', 'Workflow name', 'Workflow ID', 'State', 'Created at', 'Updated at'])
    self.assertEqual(1, len(wf_execs))
    self.assertEqual(wf_ex1_id, wf_execs[0]['ID'])
    self.assertEqual('a', wf_execs[0]['Description'])