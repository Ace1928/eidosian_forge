from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_name_uniqueness(self):
    self.workflow_create(self.wf_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-create', params='{0}'.format(self.wf_def))
    self.workflow_create(self.wf_def, admin=False)
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workflow-create', params='{0}'.format(self.wf_def))