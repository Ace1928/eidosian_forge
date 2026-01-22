import uuid
import fixtures
from unittest import mock
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import utils
def test_sign_cache_data(self):
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'mac', 'memcache_secret_key': 'mysecret'}
    token = uuid.uuid4().hex.encode()
    data = uuid.uuid4().hex
    token_cache = self.create_simple_middleware(conf=conf)._token_cache
    token_cache.initialize({})
    token_cache.set(token, data)
    self.assertEqual(token_cache.get(token), data)