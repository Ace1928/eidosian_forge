from saharaclient.api import images
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_images_list(self):
    url = self.URL + '/images'
    self.responses.get(url, json={'images': [self.body]})
    resp = self.client.images.list()
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp[0], images.Image)
    self.assertFields(self.body, resp[0])