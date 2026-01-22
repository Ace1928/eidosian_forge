from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_member_delete_basic(self):
    self.start_server()
    output = self.load_data(share_image=True)
    path = '/v2/images/%s/members/%s' % (output['image_id'], output['member_id'])
    response = self.api_delete(path)
    self.assertEqual(204, response.status_code)
    response = self.api_get(path)
    self.assertEqual(404, response.status_code)
    self.set_policy_rules({'delete_member': '!', 'add_member': '@', 'get_image': '@'})
    add_path = '/v2/images/%s/members' % output['image_id']
    data = {'member': uuids.random_member}
    response = self.api_post(add_path, json=data)
    self.assertEqual(200, response.status_code)
    response = self.api_delete(path)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'delete_member': '!', 'get_image': '!', 'get_member': '@'})
    response = self.api_delete(path)
    self.assertEqual(404, response.status_code)