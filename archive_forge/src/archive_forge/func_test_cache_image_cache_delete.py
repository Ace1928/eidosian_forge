import time
from unittest import mock
import oslo_policy.policy
from glance.api import policy
from glance.tests import functional
def test_cache_image_cache_delete(self):
    self.start_server(enable_cache=True)
    images = self.load_data()
    output = self.list_cache()
    self.assertEqual(0, len(output['queued_images']))
    self.assertEqual(0, len(output['cached_images']))
    self.cache_queue(images['public'])
    self.wait_for_caching(images['public'])
    output = self.list_cache()
    self.assertEqual(0, len(output['queued_images']))
    self.assertEqual(1, len(output['cached_images']))
    self.assertIn(images['public'], output['cached_images'][0]['image_id'])
    path = '/v2/images/%s' % images['public']
    response = self.api_delete(path)
    self.assertEqual(204, response.status_code)
    output = self.list_cache()
    self.assertEqual(0, len(output['queued_images']))
    self.assertEqual(0, len(output['cached_images']))