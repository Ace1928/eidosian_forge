from openstack import exceptions
from openstack.tests.functional import base
def test_create_application_credentials(self):
    app_creds = self._create_application_credentials()
    self.assertEqual(app_creds['user_id'], self.user_id)