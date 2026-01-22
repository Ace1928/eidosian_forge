import base64
import copy
import hashlib
import jwt.utils
import logging
import ssl
from testtools import matchers
import time
from unittest import mock
import uuid
import webob.dec
import fixtures
from oslo_config import cfg
import six
from six.moves import http_client
import testresources
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from keystonemiddleware.auth_token import _cache
from keystonemiddleware import external_oauth2_token
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit.auth_token.test_auth_token_middleware \
from keystonemiddleware.tests.unit import client_fixtures
from keystonemiddleware.tests.unit import utils
def test_token_cache_factory_secure(self):
    conf = copy.deepcopy(self._test_conf)
    conf['memcache_secret_key'] = 'test_key'
    conf['memcache_security_strategy'] = 'MAC'
    self.set_middleware(conf=conf)
    self.assertIsInstance(self.middleware._token_cache, _cache.SecureTokenCache)
    conf['memcache_security_strategy'] = 'ENCRYPT'
    self.set_middleware(conf=conf)
    self.assertIsInstance(self.middleware._token_cache, _cache.SecureTokenCache)