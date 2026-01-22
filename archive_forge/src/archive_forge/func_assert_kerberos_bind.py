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
def assert_kerberos_bind(self, token, bind_level, use_kerberos=True, success=True):
    conf = {'enforce_token_bind': bind_level, 'auth_version': self.auth_version}
    self.set_middleware(conf=conf)
    req = webob.Request.blank('/')
    req.headers['X-Auth-Token'] = token
    if use_kerberos:
        if use_kerberos is True:
            req.environ['REMOTE_USER'] = self.examples.KERBEROS_BIND
        else:
            req.environ['REMOTE_USER'] = use_kerberos
        req.environ['AUTH_TYPE'] = 'Negotiate'
    resp = req.get_response(self.middleware)
    if success:
        self.assertEqual(200, resp.status_int)
        self.assertEqual(FakeApp.SUCCESS, resp.body)
        self.assertIn('keystone.token_info', req.environ)
        self.assert_valid_last_url(token)
    else:
        self.assertEqual(401, resp.status_int)
        msg = 'Keystone uri="https://keystone.example.com:1234"'
        self.assertEqual(msg, resp.headers['WWW-Authenticate'])