import copy
import ssl
import time
from unittest import mock
from dogpile.cache import proxy
from oslo_config import cfg
from oslo_utils import uuidutils
from pymemcache import KeepaliveOpts
from oslo_cache import _opts
from oslo_cache import core as cache
from oslo_cache import exception
from oslo_cache.tests import test_cache
def test_cache_fallthrough_expiration_time_fn(self):
    self._add_dummy_config_group()
    cache_time = 599
    expiration_time = cache._get_expiration_time_fn(self.config_fixture.conf, TEST_GROUP)
    do_test = self._get_cache_fallthrough_fn(cache_time)
    self.config_fixture.config(cache_time=None, group=TEST_GROUP)
    test_value = TestProxyValue(uuidutils.generate_uuid(dashed=False))
    self.assertIsNone(expiration_time())
    do_test(value=test_value)