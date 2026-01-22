from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
def test_image_set_data_unknown_size(self):
    context = glance.context.RequestContext(user=USER1)
    image_stub = ImageStub(UUID2, status='queued', locations=[])
    image_stub.disk_format = 'iso'
    image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
    image.set_data('YYYY', None)
    self.assertEqual(4, image.size)
    self.assertEqual(UUID2, image.locations[0]['url'])
    self.assertEqual('Z', image.checksum)
    self.assertEqual('active', image.status)
    image.delete()
    self.assertEqual(image.status, 'deleted')
    self.assertRaises(glance_store.NotFound, self.store_api.get_from_backend, image.locations[0]['url'], context={})