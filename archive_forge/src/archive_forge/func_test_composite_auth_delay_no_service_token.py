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
def test_composite_auth_delay_no_service_token(self):
    self.middleware._delay_auth_decision = True
    self.purge_service_token_expected_env()
    req = webob.Request.blank('/')
    req.headers['X-Auth-Token'] = self.token_dict['uuid_token_default']
    for key, value in self.service_token_expected_env.items():
        header_key = key[len('HTTP_'):].replace('_', '-')
        req.headers[header_key] = value
    req.headers['X-Foo'] = 'Bar'
    resp = req.get_response(self.middleware)
    for key in self.service_token_expected_env.keys():
        header_key = key[len('HTTP_'):].replace('_', '-')
        self.assertFalse(req.headers.get(header_key))
    self.assertEqual('Bar', req.headers.get('X-Foo'))
    self.assertEqual(418, resp.status_int)
    self.assertEqual(FakeApp.FORBIDDEN, resp.body)