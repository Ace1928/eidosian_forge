from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_create_public_action(self):
    act = self.action_create(self.act_def, scope='public')
    same_act = self.mistral_alt_user('action-get', params=act[0]['Name'])
    self.assertEqual(act[0]['Name'], self.get_field_value(same_act, 'Name'))