from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_stage(self):
    self.start_server()
    self._create_and_stage(expected_code=204)
    self.set_policy_rules({'get_image': '!', 'modify_image': '', 'add_image': ''})
    self._create_and_stage(expected_code=204)
    self.set_policy_rules({'get_image': '', 'modify_image': '!', 'add_image': ''})
    self._create_and_stage(expected_code=403)
    self.set_policy_rules({'get_image': '!', 'modify_image': '!', 'add_image': ''})
    self._create_and_stage(expected_code=404)
    self.set_policy_rules({'get_image': '', 'modify_image': '!', 'add_image': '', 'add_member': ''})
    resp = self.api_post('/v2/images', json={'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw', 'visibility': 'shared'})
    self.assertEqual(201, resp.status_code, resp.text)
    image = resp.json
    headers = self._headers({'X-Project-Id': 'fake-tenant-id', 'Content-Type': 'application/octet-stream'})
    resp = self.api_put('/v2/images/%s/stage' % image['id'], headers=headers, data=b'IMAGEDATA')
    self.assertEqual(404, resp.status_code)
    path = '/v2/images/%s/members' % image['id']
    data = {'member': uuids.random_member}
    response = self.api_post(path, json=data)
    member = response.json
    self.assertEqual(200, response.status_code)
    self.assertEqual(image['id'], member['image_id'])
    headers = self._headers({'X-Project-Id': uuids.random_member, 'X-Roles': 'member', 'Content-Type': 'application/octet-stream'})
    resp = self.api_put('/v2/images/%s/stage' % image['id'], headers=headers, data=b'IMAGEDATA')
    self.assertEqual(403, resp.status_code)