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
def test_gate_caching_iter_good_checksum(self):
    image = b'12345678990abcdefghijklmnop'
    image_id = 123
    md5 = secretutils.md5(usedforsecurity=False)
    md5.update(image)
    checksum = md5.hexdigest()
    with mock.patch('glance.db.get_api') as mock_get_db:
        db = unit_test_utils.FakeDB(initialize=False)
        mock_get_db.return_value = db
        cache = image_cache.ImageCache()
    img_iter = cache.get_caching_iter(image_id, checksum, [image])
    for chunk in img_iter:
        pass
    self.assertTrue(cache.is_cached(image_id))