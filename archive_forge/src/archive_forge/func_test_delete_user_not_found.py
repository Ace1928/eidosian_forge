import uuid
import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_delete_user_not_found(self):
    self.register_uris([dict(method='GET', uri=self._get_keystone_mock_url(resource='users'), status_code=200, json={'users': []})])
    self.assertFalse(self.cloud.delete_user(self.getUniqueString()))