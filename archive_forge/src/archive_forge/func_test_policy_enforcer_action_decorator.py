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
def test_policy_enforcer_action_decorator(self):
    action = 'example:allowed'

    @self.flask_blueprint.route('')
    @self.enforcer.policy_enforcer_action(action)
    def nothing_interesting():
        return ('OK', 200)
    self._register_blueprint_to_app()
    with self.test_client() as c:
        c.get('%s' % self.url_prefix)
        self.assertEqual(action, getattr(flask.g, self.enforcer.ACTION_STORE_ATTR))