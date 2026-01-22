import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_create_delete(self):
    execution = self.mistral_admin('execution-create', params='{0} -d "execution test"'.format(self.direct_wf['Name']))
    exec_id = self.get_field_value(execution, 'ID')
    self.assertTableStruct(execution, ['Field', 'Value'])
    wf_name = self.get_field_value(execution, 'Workflow name')
    wf_id = self.get_field_value(execution, 'Workflow ID')
    created_at = self.get_field_value(execution, 'Created at')
    description = self.get_field_value(execution, 'Description')
    self.assertEqual(self.direct_wf['Name'], wf_name)
    self.assertIsNotNone(wf_id)
    self.assertIsNotNone(created_at)
    self.assertEqual('execution test', description)
    execs = self.mistral_admin('execution-list')
    self.assertIn(exec_id, [ex['ID'] for ex in execs])
    self.assertIn(wf_name, [ex['Workflow name'] for ex in execs])
    params = '{} --force'.format(exec_id)
    self.mistral_admin('execution-delete', params=params)