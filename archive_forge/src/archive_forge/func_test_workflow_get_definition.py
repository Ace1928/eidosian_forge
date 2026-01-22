import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_get_definition(self):
    wf = self.workflow_create(self.wf_def)
    wf_name = wf[0]['Name']
    definition = self.mistral_admin('workflow-get-definition', params=wf_name)
    self.assertNotIn('404 Not Found', definition)