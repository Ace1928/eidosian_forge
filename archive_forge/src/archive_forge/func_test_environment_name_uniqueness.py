from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_environment_name_uniqueness(self):
    self.environment_create(self.env_file)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-create', params=self.env_file)
    self.environment_create(self.env_file, admin=False)
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'environment-create', params=self.env_file)