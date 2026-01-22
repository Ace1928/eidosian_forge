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
def test_user_plugin_token_properties(self):
    token = self.examples.v3_UUID_TOKEN_DEFAULT
    token_data = self.examples.TOKEN_RESPONSES[token]
    service = self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT
    service_data = self.examples.TOKEN_RESPONSES[service]
    resp = self.call_middleware(headers={'X-Service-Catalog': '[]', 'X-Auth-Token': token, 'X-Service-Token': service})
    self.assertEqual(FakeApp.SUCCESS, resp.body)
    token_auth = resp.request.environ['keystone.token_auth']
    self.assertTrue(token_auth.has_user_token)
    self.assertTrue(token_auth.has_service_token)
    self.assertEqual(token_data.user_id, token_auth.user.user_id)
    self.assertEqual(token_data.project_id, token_auth.user.project_id)
    self.assertEqual(token_data.user_domain_id, token_auth.user.user_domain_id)
    self.assertEqual(token_data.project_domain_id, token_auth.user.project_domain_id)
    self.assertThat(token_auth.user.role_names, matchers.HasLength(2))
    self.assertIn('role1', token_auth.user.role_names)
    self.assertIn('role2', token_auth.user.role_names)
    self.assertIsNone(token_auth.user.trust_id)
    self.assertEqual(service_data.user_id, token_auth.service.user_id)
    self.assertEqual(service_data.project_id, token_auth.service.project_id)
    self.assertEqual(service_data.user_domain_id, token_auth.service.user_domain_id)
    self.assertEqual(service_data.project_domain_id, token_auth.service.project_domain_id)
    self.assertThat(token_auth.service.role_names, matchers.HasLength(2))
    self.assertIn('service', token_auth.service.role_names)
    self.assertIn('service_role2', token_auth.service.role_names)
    self.assertIsNone(token_auth.service.trust_id)