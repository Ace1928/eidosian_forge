import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_execution_list(self):
    act_execs = self.parser.listing(self.mistral('action-execution-list'))
    self.assertTableStruct(act_execs, ['ID', 'Name', 'Workflow name', 'State', 'Accepted'])