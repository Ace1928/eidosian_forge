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
def test_cache_dictionary_config_builder(self):
    """Validate we build a sane dogpile.cache dictionary config."""
    self.config_fixture.config(group='cache', config_prefix='test_prefix', backend='oslo_cache.dict', expiration_time=86400, backend_argument=['arg1:test', 'arg2:test:test', 'arg3.invalid'])
    config_dict = cache._build_cache_config(self.config_fixture.conf)
    self.assertEqual(self.config_fixture.conf.cache.backend, config_dict['test_prefix.backend'])
    self.assertEqual(self.config_fixture.conf.cache.expiration_time, config_dict['test_prefix.expiration_time'])
    self.assertEqual('test', config_dict['test_prefix.arguments.arg1'])
    self.assertEqual('test:test', config_dict['test_prefix.arguments.arg2'])
    self.assertNotIn('test_prefix.arguments.arg3', config_dict)