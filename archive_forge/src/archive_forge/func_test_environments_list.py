import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_environments_list(self):
    envs = self.parser.listing(self.mistral('environment-list'))
    self.assertTableStruct(envs, ['Name', 'Description', 'Scope', 'Created at', 'Updated at'])