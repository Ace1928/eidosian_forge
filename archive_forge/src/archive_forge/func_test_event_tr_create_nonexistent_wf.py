import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_event_tr_create_nonexistent_wf(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'event-trigger-create', params='456 4307362e-4a4a-4021-aa58-0fab23c9c751 exchange topic event {} ')