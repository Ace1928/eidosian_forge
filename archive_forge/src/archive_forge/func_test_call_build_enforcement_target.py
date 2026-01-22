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
def test_call_build_enforcement_target(self):
    assertIn = self.assertIn
    assertEq = self.assertEqual
    ref_uuid = uuid.uuid4().hex

    def _enforce_mock_func(credentials, action, target, do_raise=True):
        assertIn('target.domain.id', target)
        assertEq(target['target.domain.id'], ref_uuid)

    def _build_enforcement_target():
        return {'domain': {'id': ref_uuid}}
    self.useFixture(fixtures.MockPatchObject(self.enforcer, '_enforce', _enforce_mock_func))
    argument_id = uuid.uuid4().hex
    with self.test_client() as c:
        path = '/v3/auth/tokens'
        body = self._auth_json()
        r = c.post(path, json=body, follow_redirects=True, expected_status_code=201)
        token_id = r.headers['X-Subject-Token']
        c.get('%s/argument/%s' % (self.restful_api_url_prefix, argument_id), headers={'X-Auth-Token': token_id})
        self.enforcer.enforce_call(action='example:allowed', build_target=_build_enforcement_target)