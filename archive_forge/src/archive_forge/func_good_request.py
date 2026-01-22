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
def good_request(self, app):
    admin_token_id = uuid.uuid4().hex
    admin_token = fixture.V3Token(project_id=self.project_id)
    s = admin_token.add_service('identity', name='keystone')
    s.add_standard_endpoints(internal=self.KEYSTONE_URL)
    self.requests_mock.post('%s/v3/auth/tokens' % self.AUTH_URL, json=admin_token, headers={'X-Subject-Token': admin_token_id})
    user_token_id = uuid.uuid4().hex
    user_token = fixture.V3Token()
    user_token.set_project_scope()
    request_headers = {'X-Subject-Token': user_token_id, 'X-Auth-Token': admin_token_id}
    self.requests_mock.get('%s/v3/auth/tokens' % self.KEYSTONE_BASE_URL, request_headers=request_headers, json=user_token, headers={'X-Subject-Token': uuid.uuid4().hex})
    resp = self.call(app, headers={'X-Auth-Token': user_token_id})
    return resp