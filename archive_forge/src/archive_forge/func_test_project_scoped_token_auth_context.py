import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def test_project_scoped_token_auth_context(self):
    project_scoped_token = self.get_scoped_token()
    req = self._middleware_request(project_scoped_token)
    self.assertEqual(self.project['id'], req.environ.get(authorization.AUTH_CONTEXT_ENV)['project_id'])