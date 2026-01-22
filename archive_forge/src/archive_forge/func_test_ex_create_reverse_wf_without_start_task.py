import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_ex_create_reverse_wf_without_start_task(self):
    wf = self.workflow_create(self.wf_def)
    self.create_file('input', '{\n    "farewell": "Bye"\n}\n')
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create ', params=wf[1]['Name'])