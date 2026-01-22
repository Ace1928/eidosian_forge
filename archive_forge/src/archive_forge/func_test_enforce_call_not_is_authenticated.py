from unittest import mock
import uuid
import fixtures
import flask
from flask import blueprints
import flask_restful
from oslo_policy import policy
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import rest
def test_enforce_call_not_is_authenticated(self):
    with self.test_client() as c:
        c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex))
        with mock.patch.object(self.enforcer, '_get_oslo_req_context', return_value=None):
            self.assertRaises(exception.Unauthorized, self.enforcer.enforce_call, action='example:allowed')
        ctx = self.enforcer._get_oslo_req_context()
        ctx.authenticated = False
        self.assertRaises(exception.Unauthorized, self.enforcer.enforce_call, action='example:allowed')