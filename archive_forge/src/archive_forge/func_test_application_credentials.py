from openstack import exceptions
from openstack.tests.functional import base
def test_application_credentials(self):
    self._create_application_credentials()
    app_creds = self.conn.identity.application_credentials(user=self.user_id)
    for app_cred in app_creds:
        self.assertEqual(app_cred['user_id'], self.user_id)