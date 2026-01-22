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
def test_app_cred_matching_rules(self):
    self.set_middleware(conf={'service_type': 'compute'})
    token = self.examples.v3_APP_CRED_MATCHING_RULES
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v2.1/servers/foobar')
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2.1/servers/foobar/barfoo')
    self.set_middleware(conf={'service_type': 'image'})
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v2/images/foobar')
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2/images/foobar/barfoo')
    self.set_middleware(conf={'service_type': 'identity'})
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v3/projects/123/users/456/roles/member')
    self.set_middleware(conf={'service_type': 'block-storage'})
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v3/123/types/456')
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v3/123/types')
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2/123/types/456')
    self.set_middleware(conf={'service_type': 'object-store'})
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=200, method='GET', path='/v1/1/2/3')
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v1/1/2')
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/v2/1/2')
    self.call_middleware(headers={'X-Auth-Token': token}, expected_status=401, method='GET', path='/info')