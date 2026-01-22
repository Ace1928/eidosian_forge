import uuid
import fixtures
from unittest import mock
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import utils
def test_assert_valid_memcache_protection_config(self):
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'Encrypt'}
    self.assertRaises(exc.ConfigurationError, self.create_simple_middleware, conf=conf)
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'whatever'}
    self.assertRaises(exc.ConfigurationError, self.create_simple_middleware, conf=conf)
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'mac'}
    self.assertRaises(exc.ConfigurationError, self.create_simple_middleware, conf=conf)
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'Encrypt', 'memcache_secret_key': ''}
    self.assertRaises(exc.ConfigurationError, self.create_simple_middleware, conf=conf)
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'mAc', 'memcache_secret_key': ''}
    self.assertRaises(exc.ConfigurationError, self.create_simple_middleware, conf=conf)