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
def test_prune(self):
    """
        Test that pruning the cache works as expected...
        """
    self.assertEqual(0, self.cache.get_cache_size())
    for x in range(10):
        FIXTURE_FILE = io.BytesIO(FIXTURE_DATA)
        self.assertTrue(self.cache.cache_image_file(x, FIXTURE_FILE))
    self.assertEqual(10 * units.Ki, self.cache.get_cache_size())
    for x in range(10):
        buff = io.BytesIO()
        with self.cache.open_for_read(x) as cache_file:
            for chunk in cache_file:
                buff.write(chunk)
    FIXTURE_FILE = io.BytesIO(FIXTURE_DATA)
    self.assertTrue(self.cache.cache_image_file(99, FIXTURE_FILE))
    self.cache.prune()
    self.assertEqual(5 * units.Ki, self.cache.get_cache_size())
    for x in range(0, 6):
        self.assertFalse(self.cache.is_cached(x), 'Image %s was cached!' % x)
    for x in range(6, 10):
        self.assertTrue(self.cache.is_cached(x), 'Image %s was not cached!' % x)
    self.assertTrue(self.cache.is_cached(99), 'Image 99 was not cached!')