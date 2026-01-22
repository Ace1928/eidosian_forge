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
def assert_valid_request_200(self, token, with_catalog=True):
    resp = self.call_middleware(headers={'X-Auth-Token': token})
    if with_catalog:
        self.assertTrue(resp.request.headers.get('X-Service-Catalog'))
    else:
        self.assertNotIn('X-Service-Catalog', resp.request.headers)
    self.assertEqual(FakeApp.SUCCESS, resp.body)
    self.assertIn('keystone.token_info', resp.request.environ)
    return resp.request