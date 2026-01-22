from openstack import exceptions
from openstack.tests.functional import base
def test_delete_access_rule(self):
    app_cred = self._create_application_credential_with_access_rule()
    access_rule_id = app_cred['access_rules'][0]['id']
    self.assertRaises(exceptions.HttpException, self.conn.identity.delete_access_rule, user=self.user_id, access_rule=access_rule_id)
    self.conn.identity.delete_application_credential(user=self.user_id, application_credential=app_cred['id'])
    self.conn.identity.delete_access_rule(user=self.user_id, access_rule=access_rule_id)