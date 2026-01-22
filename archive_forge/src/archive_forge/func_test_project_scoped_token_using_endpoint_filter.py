import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_project_scoped_token_using_endpoint_filter(self):
    """Verify endpoints from project scoped token filtered."""
    ref = unit.new_project_ref(domain_id=self.domain_id)
    r = self.post('/projects', body={'project': ref})
    project = self.assertValidProjectResponse(r, ref)
    self.put('/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'project_id': project['id'], 'role_id': self.role['id']})
    body = {'user': {'default_project_id': project['id']}}
    r = self.patch('/users/%(user_id)s' % {'user_id': self.user['id']}, body=body)
    self.assertValidUserResponse(r)
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': project['id'], 'endpoint_id': self.endpoint_id})
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'])
    r = self.post('/auth/tokens', body=auth_data)
    self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
    self.assertEqual(project['id'], r.result['token']['project']['id'])