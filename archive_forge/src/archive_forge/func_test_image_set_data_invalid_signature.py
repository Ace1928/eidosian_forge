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
def test_image_set_data_invalid_signature(self):
    context = glance.context.RequestContext(user=USER1)
    extra_properties = {'img_signature_certificate_uuid': 'UUID', 'img_signature_hash_method': 'METHOD', 'img_signature_key_type': 'TYPE', 'img_signature': 'INVALID'}
    image_stub = ImageStub(UUID2, status='queued', extra_properties=extra_properties)
    self.mock_object(signature_utils, 'get_verifier', unit_test_utils.fake_get_verifier)
    image = glance.location.ImageProxy(image_stub, context, self.store_api, self.store_utils)
    with mock.patch.object(self.store_api, 'delete_from_backend') as mock_delete:
        self.assertRaises(cursive_exception.SignatureVerificationError, image.set_data, 'YYYY', 4)
        mock_delete.assert_called()