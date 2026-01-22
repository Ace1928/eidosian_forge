import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_get(self):
    created = self.workflow_create(self.wf_def)
    wf_name = created[0]['Name']
    fetched = self.mistral_admin('workflow-get', params=wf_name)
    fetched_wf_name = self.get_field_value(fetched, 'Name')
    self.assertEqual(wf_name, fetched_wf_name)