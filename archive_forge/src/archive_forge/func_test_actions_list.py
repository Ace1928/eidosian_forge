import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_actions_list(self):
    actions = self.parser.listing(self.mistral('action-list'))
    self.assertTableStruct(actions, ['Name', 'Is system', 'Input', 'Description', 'Tags', 'Created at', 'Updated at'])