from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_delete_from_store(self):
    self.start_server()
    image_id = self._create_and_import(stores=['store1', 'store2', 'store3'])
    path = '/v2/stores/store1/%s' % image_id
    response = self.api_delete(path)
    self.assertEqual(204, response.status_code)
    self.set_policy_rules({'get_image': '', 'delete_image_location': '', 'get_image_location': '!'})
    path = '/v2/stores/store2/%s' % image_id
    response = self.api_delete(path)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'get_image': '', 'delete_image_location': '!', 'get_image_location': ''})
    path = '/v2/stores/store2/%s' % image_id
    response = self.api_delete(path)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'get_image': '!', 'delete_image_location': '!', 'get_image_location': '!'})
    path = '/v2/stores/store2/%s' % image_id
    response = self.api_delete(path)
    self.assertEqual(404, response.status_code)
    self.set_policy_rules({'get_image': '!', 'delete_image_location': '', 'get_image_location': ''})
    path = '/v2/stores/store2/%s' % image_id
    response = self.api_delete(path)
    self.assertEqual(204, response.status_code)
    self.set_policy_rules({'get_image': '', 'delete_image_location': '', 'get_image_location': ''})
    headers = self._headers({'X-Roles': 'member'})
    path = '/v2/stores/store2/%s' % image_id
    response = self.api_delete(path, headers=headers)
    self.assertEqual(403, response.status_code)