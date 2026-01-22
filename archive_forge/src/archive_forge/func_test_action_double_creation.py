import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_double_creation(self):
    self.action_create(self.act_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-create', params='{0}'.format(self.act_def))