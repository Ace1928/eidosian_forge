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
def test_cache_dictionary_config_builder_redis_with_auth(self):
    """Validate the backend is reset to default if caching is disabled."""
    self.config_fixture.config(group='cache', config_prefix='test_prefix', backend='dogpile.cache.redis', redis_server='[::1]:6379', redis_username='user', redis_password='secrete')
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertEqual('redis://user:secrete@[::1]:6379', config_dict['test_prefix.arguments.url'])