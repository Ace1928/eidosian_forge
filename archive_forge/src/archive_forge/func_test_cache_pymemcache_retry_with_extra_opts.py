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
def test_cache_pymemcache_retry_with_extra_opts(self):
    """Validate we build a valid config for the retry client."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', enable_retry_client=True, retry_attempts=42, retry_delay=42, hashclient_retry_attempts=100, hashclient_retry_delay=100, dead_timeout=100)
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertTrue(self.config_fixture.conf.cache.enable_retry_client)
    self.assertEqual(config_dict['test_prefix.arguments.retry_attempts'], 42)
    self.assertEqual(config_dict['test_prefix.arguments.retry_delay'], 42)
    self.assertEqual(config_dict['test_prefix.arguments.hashclient_retry_attempts'], 100)
    self.assertEqual(config_dict['test_prefix.arguments.hashclient_retry_delay'], 100)
    self.assertEqual(config_dict['test_prefix.arguments.dead_timeout'], 100)