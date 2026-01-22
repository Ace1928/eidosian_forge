from saharaclient.api import images
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_images_get(self):
    url = self.URL + '/images/id'
    self.responses.get(url, json={'image': self.body})
    resp = self.client.images.get('id')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, images.Image)
    self.assertFields(self.body, resp)