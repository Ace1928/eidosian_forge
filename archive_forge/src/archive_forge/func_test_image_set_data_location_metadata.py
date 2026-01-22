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
def test_image_set_data_location_metadata(self):
    context = glance.context.RequestContext(user=USER1)
    image_stub = ImageStub(UUID2, status='queued', locations=[])
    loc_meta = {'key': 'value5032'}
    store_api = unit_test_utils.FakeStoreAPI(store_metadata=loc_meta)
    store_utils = unit_test_utils.FakeStoreUtils(store_api)
    image = glance.location.ImageProxy(image_stub, context, store_api, store_utils)
    image.set_data('YYYY', 4)
    self.assertEqual(4, image.size)
    location_data = image.locations[0]
    self.assertEqual(UUID2, location_data['url'])
    self.assertEqual(loc_meta, location_data['metadata'])
    self.assertEqual('Z', image.checksum)
    self.assertEqual('active', image.status)
    image.delete()
    self.assertEqual(image.status, 'deleted')
    self.assertRaises(glance_store.NotFound, self.store_api.get_from_backend, image.locations[0]['url'], {})