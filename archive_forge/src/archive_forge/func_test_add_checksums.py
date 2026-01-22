import hashlib
import io
from unittest import mock
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils as test_utils
def test_add_checksums(self):
    self.store.chunk_size = units.Ki
    image_id = 'fake_image_id'
    file_size = 5 * units.Ki
    file_contents = b'*' * file_size
    image_file = io.BytesIO(file_contents)
    expected_checksum = md5(file_contents, usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(file_contents).hexdigest()
    with mock.patch.object(rbd_store.rbd.Image, 'write'):
        loc, size, checksum, multihash, _ = self.store.add(image_id, image_file, file_size, self.hash_algo)
    self.assertEqual(expected_checksum, checksum)
    self.assertEqual(expected_multihash, multihash)