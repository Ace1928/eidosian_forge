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
def get_admin_token(self):
    """Convenience method so that we can test authenticated requests."""
    r = self.admin_request(method='POST', path='/v3/auth/tokens', body={'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.user_reqadmin['name'], 'password': self.user_reqadmin['password'], 'domain': {'id': self.user_reqadmin['domain_id']}}}}, 'scope': {'project': {'id': self.default_domain_project_id}}}})
    return r.headers.get('X-Subject-Token')