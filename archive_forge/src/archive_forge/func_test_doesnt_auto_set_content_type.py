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
def test_doesnt_auto_set_content_type(self):
    text = uuid.uuid4().hex

    def _middleware(environ, start_response):
        start_response(200, [])
        return text

    def _start_response(status_code, headerlist, exc_info=None):
        self.assertIn('200', status_code)
        self.assertEqual([], headerlist)
    m = auth_token.AuthProtocol(_middleware, self.conf)
    env = {'REQUEST_METHOD': 'GET', 'HTTP_X_AUTH_TOKEN': self.token_dict['uuid_token_default']}
    r = m(env, _start_response)
    self.assertEqual(text, r)