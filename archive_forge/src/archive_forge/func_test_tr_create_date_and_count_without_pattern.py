import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_tr_create_date_and_count_without_pattern(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --count 42 --first-time "4242-12-25 13:37"')