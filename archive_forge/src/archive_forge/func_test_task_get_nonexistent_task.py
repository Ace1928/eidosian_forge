import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_task_get_nonexistent_task(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'task-get', params='nonexist')