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
def get_domain_scoped_token(self):
    """Convenience method for requesting domain scoped token."""
    r = self.admin_request(method='POST', path='/v3/auth/tokens', body={'auth': {'identity': {'methods': ['password'], 'password': {'user': {'name': self.user['name'], 'password': self.user['password'], 'domain': {'id': self.user['domain_id']}}}}, 'scope': {'domain': {'id': self.domain['id']}}}})
    return r.headers.get('X-Subject-Token')