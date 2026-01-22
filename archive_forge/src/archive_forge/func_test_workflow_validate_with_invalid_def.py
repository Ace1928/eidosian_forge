import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_validate_with_invalid_def(self):
    self.create_file('wf.yaml', 'name: wf\n')
    wf = self.mistral_admin('workflow-validate', params='wf.yaml')
    wf_valid = self.get_field_value(wf, 'Valid')
    wf_error = self.get_field_value(wf, 'Error')
    self.assertEqual('False', wf_valid)
    self.assertNotEqual('None', wf_error)