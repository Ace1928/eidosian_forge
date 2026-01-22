import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_delete_nonexistent_act(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-delete', params='nonexist')