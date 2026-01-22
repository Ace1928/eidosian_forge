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
def test_cache_dictionary_config_builder_tls_enabled_redis_sentinel(self):
    """Validate the backend is reset to default if caching is disabled."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.redis_sentinel', tls_enabled=True, tls_cafile='path_to_ca_file', tls_keyfile='path_to_key_file', tls_certfile='path_to_cert_file')
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertTrue(self.config_fixture.conf.cache.tls_enabled)
    self.assertIn('test_prefix.arguments.connection_kwargs', config_dict)
    self.assertEqual({'ssl': True, 'ssl_ca_certs': 'path_to_ca_file', 'ssl_keyfile': 'path_to_key_file', 'ssl_certfile': 'path_to_cert_file'}, config_dict['test_prefix.arguments.connection_kwargs'])
    self.assertIn('test_prefix.arguments.sentinel_kwargs', config_dict)
    self.assertEqual({'ssl': True, 'ssl_ca_certs': 'path_to_ca_file', 'ssl_keyfile': 'path_to_key_file', 'ssl_certfile': 'path_to_cert_file'}, config_dict['test_prefix.arguments.sentinel_kwargs'])