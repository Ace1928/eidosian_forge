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
@mock.patch('oslo_cache.core._LOG')
def test_cache_dictionary_config_builder_fips_mode_supported(self, log):
    """Validate the FIPS mode is supported."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='dogpile.cache.pymemcache', tls_enabled=True, enforce_fips_mode=True)
    with mock.patch.object(ssl, 'FIPS_mode', create=True, return_value=True):
        with mock.patch.object(ssl, 'FIPS_mode_set', create=True):
            cache._build_cache_config(self.config_fixture.conf)
            log.info.assert_called_once_with('Enforcing the use of the OpenSSL FIPS mode')