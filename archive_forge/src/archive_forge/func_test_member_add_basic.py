from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_member_add_basic(self):
    self.start_server()
    output = self.load_data()
    path = '/v2/images/%s/members' % output['image_id']
    data = {'member': uuids.random_member}
    response = self.api_post(path, json=data)
    self.assertEqual(200, response.status_code)
    member = response.json
    self.assertEqual(output['image_id'], member['image_id'])
    self.assertEqual('pending', member['status'])
    self.set_policy_rules({'add_member': '!', 'get_image': '@'})
    response = self.api_post(path, json=data)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'add_member': '!', 'get_image': '!'})
    response = self.api_post(path, json=data)
    self.assertEqual(404, response.status_code)