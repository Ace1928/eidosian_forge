import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_get_output(self):
    execution = self.execution_create(self.direct_wf['Name'])
    exec_id = self.get_field_value(execution, 'ID')
    ex_output = self.mistral_admin('execution-get-output', params=exec_id)
    self.assertEqual([], ex_output)