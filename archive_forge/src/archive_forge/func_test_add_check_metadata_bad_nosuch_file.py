import builtins
import errno
import hashlib
import io
import json
import os
import stat
from unittest import mock
import uuid
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers import filesystem
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def test_add_check_metadata_bad_nosuch_file(self):
    expected_image_id = str(uuid.uuid4())
    jsonfilename = os.path.join(self.test_dir, 'storage_metadata.%s' % expected_image_id)
    self.config(filesystem_store_metadata_file=jsonfilename, group='glance_store')
    expected_file_size = 10
    expected_file_contents = b'*' * expected_file_size
    image_file = io.BytesIO(expected_file_contents)
    location, size, checksum, multihash, metadata = self.store.add(expected_image_id, image_file, expected_file_size, self.hash_algo)
    self.assertEqual(metadata, {})