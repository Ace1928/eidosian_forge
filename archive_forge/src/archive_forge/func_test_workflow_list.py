import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_list(self):
    workflows = self.parser.listing(self.mistral('workflow-list'))
    self.assertTableStruct(workflows, ['ID', 'Name', 'Tags', 'Input', 'Scope', 'Created at', 'Updated at'])