import uuid
import fixtures
from unittest import mock
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import utils
@mock.patch('keystonemiddleware.auth_token._memcache_crypt.unprotect_data')
def test_corrupted_cache_data(self, mocked_decrypt_data):
    mocked_decrypt_data.side_effect = Exception('corrupted')
    conf = {'memcached_servers': ','.join(MEMCACHED_SERVERS), 'memcache_security_strategy': 'encrypt', 'memcache_secret_key': 'mysecret'}
    token = uuid.uuid4().hex.encode()
    data = uuid.uuid4().hex
    token_cache = self.create_simple_middleware(conf=conf)._token_cache
    token_cache.initialize({})
    token_cache.set(token, data)
    self.assertIsNone(token_cache.get(token))