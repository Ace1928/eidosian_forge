from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_delete(self):
    self.start_server()
    image_id = self._create_and_upload()
    resp = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(204, resp.status_code)
    resp = self.api_get('/v2/images/%s' % image_id)
    self.assertEqual(404, resp.status_code)
    resp = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(404, resp.status_code)
    image_id = self._create_and_upload()
    self.set_policy_rules({'get_image': '', 'delete_image': '!'})
    resp = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_image': '!', 'delete_image': '!'})
    resp = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'get_image': '!', 'delete_image': ''})
    resp = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(204, resp.status_code)