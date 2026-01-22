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
def test_loading_password_plugin(self):
    opts = loading.get_auth_plugin_conf_options('password')
    self.cfg.register_opts(opts, group=_base.AUTHTOKEN_GROUP)
    project_id = uuid.uuid4().hex
    loading.register_auth_conf_options(self.cfg.conf, group=_base.AUTHTOKEN_GROUP)
    self.cfg.config(auth_type='password', username='testuser', password='testpass', auth_url=self.AUTH_URL, project_id=project_id, user_domain_id='userdomainid', group=_base.AUTHTOKEN_GROUP)
    body = uuid.uuid4().hex
    app = self.create_simple_middleware(body=body)
    resp = self.good_request(app)
    self.assertEqual(body.encode(), resp.body)