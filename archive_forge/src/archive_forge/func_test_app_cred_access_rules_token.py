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
def test_app_cred_access_rules_token(self):
    self.set_middleware(conf={'service_type': 'compute'})
    token = self.examples.v3_APP_CRED_ACCESS_RULES
    token_data = self.examples.TOKEN_RESPONSES[token]
    resp = self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v2.1/servers')
    token_auth = resp.request.environ['keystone.token_auth']
    self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)
    self.assertEqual(token_data.application_credential_access_rules, token_auth.user.application_credential_access_rules)
    resp = self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2.1/servers/someuuid')
    token_auth = resp.request.environ['keystone.token_auth']
    self.assertEqual(token_data.application_credential_id, token_auth.user.application_credential_id)
    self.assertEqual(token_data.application_credential_access_rules, token_auth.user.application_credential_access_rules)