import http.client
from oslo_config import cfg
from oslo_serialization import jsonutils
from glance.tests.integration.v2 import base
def test_property_ops_when_quota_violated(self):
    image_list = self._get()['images']
    self.assertEqual(0, len(image_list))
    orig_property_quota = 10
    CONF.set_override('image_property_quota', orig_property_quota)
    req_body = {'name': 'testimg', 'disk_format': 'aki', 'container_format': 'aki'}
    for i in range(orig_property_quota):
        req_body['k_%d' % i] = 'v_%d' % i
    image = self._create_image(req_body)
    image_id = image['id']
    for i in range(orig_property_quota):
        self.assertEqual('v_%d' % i, image['k_%d' % i])
    self.config(image_property_quota=2)
    patch_body = [{'op': 'replace', 'path': '/k_4', 'value': 'v_4.new'}]
    image = jsonutils.loads(self._patch(image_id, patch_body, http.client.OK))
    self.assertEqual('v_4.new', image['k_4'])
    patch_body = [{'op': 'remove', 'path': '/k_7'}]
    image = jsonutils.loads(self._patch(image_id, patch_body, http.client.OK))
    self.assertNotIn('k_7', image)
    patch_body = [{'op': 'add', 'path': '/k_100', 'value': 'v_100'}]
    self._patch(image_id, patch_body, http.client.REQUEST_ENTITY_TOO_LARGE)
    image = self._get(image_id)
    self.assertNotIn('k_100', image)
    patch_body = [{'op': 'remove', 'path': '/k_5'}, {'op': 'add', 'path': '/k_100', 'value': 'v_100'}]
    self._patch(image_id, patch_body, http.client.REQUEST_ENTITY_TOO_LARGE)
    image = self._get(image_id)
    self.assertNotIn('k_100', image)
    self.assertIn('k_5', image)
    patch_body = [{'op': 'add', 'path': '/k_100', 'value': 'v_100'}, {'op': 'add', 'path': '/k_99', 'value': 'v_99'}]
    to_rm = ['k_%d' % i for i in range(orig_property_quota) if i != 7]
    patch_body.extend([{'op': 'remove', 'path': '/%s' % k} for k in to_rm])
    image = jsonutils.loads(self._patch(image_id, patch_body, http.client.OK))
    self.assertEqual('v_99', image['k_99'])
    self.assertEqual('v_100', image['k_100'])
    for k in to_rm:
        self.assertNotIn(k, image)