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
def test_oslo_context(self):
    token = self.get_scoped_token()
    request_id = uuid.uuid4().hex
    environ = {'openstack.request_id': request_id}
    self._middleware_request(token, extra_environ=environ)
    req_context = oslo_context.context.get_current()
    self.assertEqual(request_id, req_context.request_id)
    self.assertEqual(token, req_context.auth_token)
    self.assertEqual(self.user['id'], req_context.user_id)
    self.assertEqual(self.project['id'], req_context.project_id)
    self.assertIsNone(req_context.domain_id)
    self.assertEqual(self.user['domain_id'], req_context.user_domain_id)
    self.assertEqual(self.project['domain_id'], req_context.project_domain_id)
    self.assertFalse(req_context.is_admin)