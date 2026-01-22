from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_member_update_basic(self):
    self.start_server()
    output = self.load_data(share_image=True)
    path = '/v2/images/%s/members/%s' % (output['image_id'], output['member_id'])
    data = {'status': 'accepted'}
    response = self.api_put(path, json=data)
    self.assertEqual(200, response.status_code)
    member = response.json
    self.assertEqual(output['image_id'], member['image_id'])
    self.assertEqual('accepted', member['status'])
    self.set_policy_rules({'modify_member': '!', 'get_image': '@'})
    response = self.api_put(path, json=data)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'modify_member': '!', 'get_image': '!', 'get_member': '@'})
    headers = self._headers({'X-Tenant-Id': 'fake-tenant-id'})
    response = self.api_put(path, headers=headers, json=data)
    self.assertEqual(404, response.status_code)