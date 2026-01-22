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
def test_auth_plugin(self):
    for service_url in (self.examples.UNVERSIONED_SERVICE_URL, self.examples.SERVICE_URL):
        self.requests_mock.get(service_url, json=VERSION_LIST_v3, status_code=300)
    token = self.token_dict['uuid_token_default']
    resp = self.call_middleware(headers={'X-Auth-Token': token})
    self.assertEqual(FakeApp.SUCCESS, resp.body)
    token_auth = resp.request.environ['keystone.token_auth']
    endpoint_filter = {'service_type': self.examples.SERVICE_TYPE, 'version': 3}
    url = token_auth.get_endpoint(session.Session(), **endpoint_filter)
    self.assertEqual('%s/v3' % BASE_URI, url)
    self.assertTrue(token_auth.has_user_token)
    self.assertFalse(token_auth.has_service_token)
    self.assertIsNone(token_auth.service)