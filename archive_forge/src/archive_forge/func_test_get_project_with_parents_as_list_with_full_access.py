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
def test_get_project_with_parents_as_list_with_full_access(self):
    """``GET /projects/{project_id}?parents_as_list`` with full access.

        Test plan:

        - Create 'parent', 'project' and 'subproject' projects;
        - Assign a user a role on each one of those projects;
        - Check that calling parents_as_list on 'subproject' returns both
          'project' and 'parent'.

        """
    parent, project, subproject = self._create_projects_hierarchy(2)
    for proj in (parent, project, subproject):
        self.put(self.build_role_assignment_link(role_id=self.role_id, user_id=self.user_id, project_id=proj['project']['id']))
    r = self.get('/projects/%(project_id)s?parents_as_list' % {'project_id': subproject['project']['id']})
    self.assertValidProjectResponse(r, subproject['project'])
    self.assertIn(project, r.result['project']['parents'])
    self.assertIn(parent, r.result['project']['parents'])
    self.assertEqual(2, len(r.result['project']['parents']))