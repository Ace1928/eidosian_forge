import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def workflow_member_create(self, wf_id):
    cmd_param = '%s workflow %s' % (wf_id, self.get_project_id('alt_demo'))
    member = self.mistral_admin('member-create', params=cmd_param)
    self.addCleanup(self.mistral_admin, 'member-delete', params=cmd_param)
    return member