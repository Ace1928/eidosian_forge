import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_get_nonexistent(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral, 'action-get', params='nonexist')