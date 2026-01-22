import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_get_definition_with_namespace(self):
    self.action_create(self.act_def)
    definition = self.mistral_admin('action-get-definition', params='greeting --namespace test_namespace')
    self.assertNotIn('404 Not Found', definition)