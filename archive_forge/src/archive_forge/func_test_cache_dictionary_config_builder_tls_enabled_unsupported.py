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
def test_cache_dictionary_config_builder_tls_enabled_unsupported(self):
    """Validate the tls_enabled opiton is not supported.."""
    self.config_fixture.config(group='cache', enabled=True, config_prefix='test_prefix', backend='oslo_cache.dict', tls_enabled=True)
    with mock.patch.object(ssl, 'create_default_context'):
        self.assertRaises(exception.ConfigurationError, cache._build_cache_config, self.config_fixture.conf)
        ssl.create_default_context.assert_not_called()