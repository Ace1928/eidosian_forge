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
def test_prune_to_zero(self):
    """Test that an image_cache_max_size of 0 doesn't kill the pruner

        This is a test specifically for LP #1039854
        """
    self.assertEqual(0, self.cache.get_cache_size())
    FIXTURE_FILE = io.BytesIO(FIXTURE_DATA)
    self.assertTrue(self.cache.cache_image_file('xxx', FIXTURE_FILE))
    self.assertEqual(1024, self.cache.get_cache_size())
    buff = io.BytesIO()
    with self.cache.open_for_read('xxx') as cache_file:
        for chunk in cache_file:
            buff.write(chunk)
    self.config(image_cache_max_size=0)
    self.cache.prune()
    self.assertEqual(0, self.cache.get_cache_size())
    self.assertFalse(self.cache.is_cached('xxx'))