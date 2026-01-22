from unittest import mock
import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import users
def test_update_password_with_no_hardcoded_endpoint_filter(self):
    old_password = uuid.uuid4().hex
    new_password = uuid.uuid4().hex
    expected_params = {'user': {'password': new_password, 'original_password': old_password}}
    user_password_update_path = '/users/%s/password' % self.TEST_USER_ID
    self.client.user_id = self.TEST_USER_ID
    with mock.patch('keystoneclient.base.Manager._update') as m:
        self.manager.update_password(old_password, new_password)
        m.assert_called_with(user_password_update_path, expected_params, method='POST', log=False)