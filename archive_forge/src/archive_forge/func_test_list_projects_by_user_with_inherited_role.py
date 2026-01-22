import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_list_projects_by_user_with_inherited_role(self):
    """Ensure the cache is invalidated when creating/deleting a project."""
    domain_ref = unit.new_domain_ref()
    resp = self.post('/domains', body={'domain': domain_ref})
    domain = resp.result['domain']
    user_ref = unit.new_user_ref(domain_id=self.domain_id)
    resp = self.post('/users', body={'user': user_ref})
    user = resp.result['user']
    role_ref = unit.new_role_ref()
    resp = self.post('/roles', body={'role': role_ref})
    role = resp.result['role']
    self.put('/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles/%(role_id)s/inherited_to_projects' % {'domain_id': domain['id'], 'user_id': user['id'], 'role_id': role['id']})
    resp = self.get('/users/%(user)s/projects' % {'user': user['id']})
    self.assertValidProjectListResponse(resp)
    self.assertEqual([], resp.result['projects'])
    project_ref = unit.new_project_ref(domain_id=domain['id'])
    resp = self.post('/projects', body={'project': project_ref})
    project = resp.result['project']
    resp = self.get('/users/%(user)s/projects' % {'user': user['id']})
    self.assertValidProjectListResponse(resp)
    self.assertEqual(project['id'], resp.result['projects'][0]['id'])