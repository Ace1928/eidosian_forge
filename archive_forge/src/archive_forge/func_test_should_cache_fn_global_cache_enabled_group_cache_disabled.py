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
def test_should_cache_fn_global_cache_enabled_group_cache_disabled(self):
    cacheable_function = self._get_cacheable_function()
    self._add_test_caching_option()
    self.config_fixture.config(group='cache', enabled=True)
    self.config_fixture.config(group='cache', caching=False)
    cacheable_function(self.test_value)
    cached_value = cacheable_function(self.test_value)
    self.assertFalse(cached_value.cached)