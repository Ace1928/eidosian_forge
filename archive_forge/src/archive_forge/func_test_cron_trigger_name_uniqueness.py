from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_cron_trigger_name_uniqueness(self):
    wf = self.workflow_create(self.wf_def)
    self.cron_trigger_create('admin_trigger', wf[0]['ID'], '{}', '5 * * * *')
    self.assertRaises(exceptions.CommandFailed, self.cron_trigger_create, 'admin_trigger', wf[0]['ID'], '{}5 * * * *')
    wf = self.workflow_create(self.wf_def, admin=False)
    self.cron_trigger_create('user_trigger', wf[0]['ID'], '{}', '5 * * * *', None, None, admin=False)
    self.assertRaises(exceptions.CommandFailed, self.cron_trigger_create, 'user_trigger', wf[0]['ID'], '{}', '5 * * * *', None, None, admin=False)