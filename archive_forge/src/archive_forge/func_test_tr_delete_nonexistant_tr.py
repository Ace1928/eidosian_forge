import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_tr_delete_nonexistant_tr(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-delete', params='tr')