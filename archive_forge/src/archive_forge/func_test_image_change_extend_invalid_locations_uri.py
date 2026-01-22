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
def test_image_change_extend_invalid_locations_uri(self):
    self.assertEqual(2, len(self.store_api.data.keys()))
    context = glance.context.RequestContext(user=USER1)
    image1, image_stub1 = self._add_image(context, UUID2, 'XXXX', 4)
    location_bad = {'url': 'unknown://location', 'metadata': {}}
    self.assertRaises(exception.BadStoreUri, image1.locations.extend, [location_bad])
    image1.delete()
    self.assertEqual(2, len(self.store_api.data.keys()))
    self.assertNotIn(UUID2, self.store_api.data.keys())