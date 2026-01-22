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
def test_client_issuer_not_trusted(self):
    env = {}
    env['SSL_CLIENT_I_DN'] = self.untrusted_client_issuer
    env['HTTP_X_PROJECT_ID'] = uuid.uuid4().hex
    req = self._do_middleware_request(extra_environ=env)
    context = req.environ.get(authorization.AUTH_CONTEXT_ENV)
    self.assertIsNone(context)