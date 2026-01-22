import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def environment_create(self, params, admin=True):
    env = self.mistral_cli(admin, 'environment-create', params=params)
    env_name = self.get_field_value(env, 'Name')
    self.addCleanup(self.mistral_cli, admin, 'environment-delete', params=env_name)
    return env