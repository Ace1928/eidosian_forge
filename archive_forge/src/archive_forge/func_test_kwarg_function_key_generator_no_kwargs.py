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
def test_kwarg_function_key_generator_no_kwargs(self):
    cacheable_function = self._get_cacheable_function(region=self.region_kwargs)
    self.config_fixture.config(group='cache', enabled=True)
    cacheable_function(self.test_value)
    cached_value = cacheable_function(self.test_value)
    self.assertTrue(cached_value.cached)