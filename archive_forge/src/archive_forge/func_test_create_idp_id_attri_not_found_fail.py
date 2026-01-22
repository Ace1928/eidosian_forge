import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
def test_create_idp_id_attri_not_found_fail(self):
    env = {}
    env[uuid.uuid4().hex] = self.client_issuer
    auth = tokenless_auth.TokenlessAuthHelper(env)
    expected_msg = 'Could not determine Identity Provider ID. The configuration option %s was not found in the request environment.' % CONF.tokenless_auth.issuer_attribute
    self.assertRaisesRegex(exception.TokenlessAuthConfigError, expected_msg, auth._build_idp_id)