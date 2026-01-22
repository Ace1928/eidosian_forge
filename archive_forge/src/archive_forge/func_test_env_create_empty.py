import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_env_create_empty(self):
    self.create_file('env.yaml')
    self.assertRaises(exceptions.CommandFailed, self.environment_create, 'env.yaml')