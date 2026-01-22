from openstack import exceptions
from openstack.tests.functional import base
def test_get_access_rule(self):
    app_cred = self._create_application_credential_with_access_rule()
    access_rule_id = app_cred['access_rules'][0]['id']
    access_rule = self.conn.identity.get_access_rule(user=self.user_id, access_rule=access_rule_id)
    self.assertEqual(access_rule['id'], access_rule_id)
    self.assertEqual(access_rule['user_id'], self.user_id)