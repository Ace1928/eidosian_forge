import struct
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.tests.unit import utils
def test_sign_data(self):
    keys = self._setup_keys(b'MAC')
    sig = memcache_crypt.sign_data(keys['MAC'], b'data')
    self.assertEqual(len(sig), memcache_crypt.DIGEST_LENGTH_B64)