import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_two_cron_triggers_for_one_wf(self):
    self.cron_trigger_create('trigger1', self.wf_name, '{}', '5 * * * *')
    self.cron_trigger_create('trigger2', self.wf_name, '{}', '15 * * * *')
    triggers = self.mistral_admin('cron-trigger-list')
    self.assertIn('trigger1', [tr['Name'] for tr in triggers])
    self.assertIn('trigger2', [tr['Name'] for tr in triggers])