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
def test_enforcer_is_admin_check_without_token(self):
    with self.test_client() as c:
        c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex), headers={authorization.AUTH_TOKEN_HEADER: 'BOGUS'})
        self.assertFalse(self.enforcer._shared_admin_auth_token_set())
        c.get('%s/argument/%s' % (self.restful_api_url_prefix, uuid.uuid4().hex))
        self.assertFalse(self.enforcer._shared_admin_auth_token_set())