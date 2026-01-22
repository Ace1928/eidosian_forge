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
def test_caching_token_invalid(self):
    conf = copy.deepcopy(self._test_conf)
    self.set_middleware(conf=conf)
    self.middleware._token_cache._env_cache_name = 'cache'
    cache = _cache._FakeClient()
    self.middleware._token_cache.initialize(env={'cache': cache})
    orig_cache_set = cache.set
    cache.set = mock.Mock(side_effect=orig_cache_set)

    def mock_resp(request, context):
        return self._introspect_response(request, context, auth_method=self._auth_method, introspect_client_id=self._test_client_id, introspect_client_secret=self._test_client_secret, access_token=self._token, active=True, metadata=self._default_metadata)
    self.requests_mock.post(self._introspect_endpoint, json=mock_resp)
    self.requests_mock.get(self._auth_url, json=VERSION_LIST_v3, status_code=300)
    self.call_middleware(headers=get_authorization_header(self._token), expected_status=200, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
    self.assertThat(1, matchers.Equals(cache.set.call_count))
    self.call_middleware(headers=get_authorization_header(str(uuid.uuid4()) + '_token'), expected_status=500, method='GET', path='/vnfpkgm/v1/vnf_packages', der_client_cert=self._der_client_cert, environ={'wsgi.input': FakeWsgiInput(FakeSocket(None))})
    self._token = self.token_dict['uuid_token_default']