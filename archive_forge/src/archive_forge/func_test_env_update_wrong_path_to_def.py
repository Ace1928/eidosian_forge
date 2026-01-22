import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_env_update_wrong_path_to_def(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-update', params='env')