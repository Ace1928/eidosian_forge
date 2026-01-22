from contextlib import contextmanager
import datetime
import errno
import io
import os
import tempfile
import time
from unittest import mock
import fixtures
import glance_store as store
from oslo_config import cfg
from oslo_utils import fileutils
from oslo_utils import secretutils
from oslo_utils import units
from glance import async_
from glance.common import exception
from glance import context
from glance import gateway as glance_gateway
from glance import image_cache
from glance.image_cache import prefetcher
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
from glance.tests.utils import skip_if_disabled
from glance.tests.utils import xattr_writes_supported
@skip_if_disabled
def test_caching_iterator(self):
    """
        Test to see if the caching iterator interacts properly with the driver
        When the iterator completes going through the data the driver should
        have closed the image and placed it correctly
        """

    def consume(image_id):
        data = [b'a', b'b', b'c', b'd', b'e', b'f']
        checksum = None
        caching_iter = self.cache.get_caching_iter(image_id, checksum, iter(data))
        self.assertEqual(data, list(caching_iter))
    image_id = '1'
    self.assertFalse(self.cache.is_cached(image_id))
    consume(image_id)
    self.assertTrue(self.cache.is_cached(image_id), 'Image %s was NOT cached!' % image_id)
    incomplete_file_path = os.path.join(self.cache_dir, 'incomplete', image_id)
    invalid_file_path = os.path.join(self.cache_dir, 'invalid', image_id)
    self.assertFalse(os.path.exists(incomplete_file_path))
    self.assertFalse(os.path.exists(invalid_file_path))