import datetime
import os
import time
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import fixture
from keystoneauth1 import loading
from keystoneauth1 import session
import oslo_cache
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import pbr.version
import testresources
from testtools import matchers
import webob
import webob.dec
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.tests.unit.auth_token import base
from keystonemiddleware.tests.unit import client_fixtures
def test_memcache_set_expired(self, extra_conf={}, extra_environ={}):
    token_cache_time = 10
    conf = {'token_cache_time': '%s' % token_cache_time}
    conf.update(extra_conf)
    self.set_middleware(conf=conf)
    token = self.token_dict['uuid_token_default']
    self.call_middleware(headers={'X-Auth-Token': token})
    req = webob.Request.blank('/')
    req.headers['X-Auth-Token'] = token
    req.environ.update(extra_environ)
    now = datetime.datetime.now(datetime.timezone.utc)
    self.useFixture(TimeFixture(now))
    req.get_response(self.middleware)
    self.assertIsNotNone(self._get_cached_token(token))
    timeutils.advance_time_seconds(token_cache_time)
    self.assertIsNone(self._get_cached_token(token))