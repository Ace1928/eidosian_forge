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
def test_auth_plugin_service_token(self):
    url = 'http://test.url'
    text = uuid.uuid4().hex
    self.requests_mock.get(url, text=text)
    token = self.token_dict['uuid_token_default']
    resp = self.call_middleware(headers={'X-Auth-Token': token})
    self.assertEqual(200, resp.status_int)
    self.assertEqual(FakeApp.SUCCESS, resp.body)
    s = session.Session(auth=resp.request.environ['keystone.token_auth'])
    resp = s.get(url)
    self.assertEqual(text, resp.text)
    self.assertEqual(200, resp.status_code)
    headers = self.requests_mock.last_request.headers
    self.assertEqual(FAKE_ADMIN_TOKEN_ID, headers['X-Service-Token'])