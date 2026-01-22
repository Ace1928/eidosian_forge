import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_validate_with_valid_def(self):
    wf = self.mistral_admin('workflow-validate', params=self.wf_def)
    wf_valid = self.get_field_value(wf, 'Valid')
    wf_error = self.get_field_value(wf, 'Error')
    self.assertEqual('True', wf_valid)
    self.assertEqual('None', wf_error)