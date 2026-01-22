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
def test_open_for_read(self):
    """Test convenience wrapper for opening a cache file via
        its image identifier.
        """
    self._setup_fixture_file()
    buff = io.BytesIO()
    with self.cache.open_for_read(1) as cache_file:
        for chunk in cache_file:
            buff.write(chunk)
    self.assertEqual(FIXTURE_DATA, buff.getvalue())