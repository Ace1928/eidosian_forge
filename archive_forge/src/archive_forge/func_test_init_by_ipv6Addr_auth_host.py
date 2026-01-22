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
def test_init_by_ipv6Addr_auth_host(self):
    del self.conf['identity_uri']
    conf = {'auth_host': '2001:2013:1:f101::1', 'auth_port': '1234', 'auth_protocol': 'http', 'www_authenticate_uri': None, 'auth_version': 'v3.0'}
    middleware = self.create_simple_middleware(conf=conf)
    self.assertEqual('http://[2001:2013:1:f101::1]:1234', middleware._www_authenticate_uri)