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
def test_app_cred_access_rules_service_request(self):
    self.set_middleware(conf={'service_type': 'image'})
    token = self.examples.v3_APP_CRED_ACCESS_RULES
    headers = {'X-Auth-Token': token}
    self.call_middleware(headers=headers, expected_status=401, method='GET', path='/v2/images')
    service_token = self.examples.v3_UUID_SERVICE_TOKEN_DEFAULT
    headers['X-Service-Token'] = service_token
    self.call_middleware(headers=headers, expected_status=200, method='GET', path='/v2/images')