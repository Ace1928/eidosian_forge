import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_execution_list_with_limit(self):
    act_execs = self.parser.listing(self.mistral('action-execution-list', params='--limit 1'))
    self.assertEqual(1, len(act_execs))