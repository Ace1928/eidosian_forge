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
def test_get_project_with_subtree_as_ids(self):
    """Call ``GET /projects/{project_id}?subtree_as_ids``.

        This test creates a more complex hierarchy to test if the structured
        dictionary returned by using the ``subtree_as_ids`` query param
        correctly represents the hierarchy.

        The hierarchy contains 5 projects with the following structure::

                                  +--A--+
                                  |     |
                               +--B--+  C
                               |     |
                               D     E


        """
    projects = self._create_projects_hierarchy(hierarchy_size=2)
    new_ref = unit.new_project_ref(domain_id=self.domain_id, parent_id=projects[0]['project']['id'])
    resp = self.post('/projects', body={'project': new_ref})
    self.assertValidProjectResponse(resp, new_ref)
    projects.append(resp.result)
    new_ref = unit.new_project_ref(domain_id=self.domain_id, parent_id=projects[1]['project']['id'])
    resp = self.post('/projects', body={'project': new_ref})
    self.assertValidProjectResponse(resp, new_ref)
    projects.append(resp.result)
    r = self.get('/projects/%(project_id)s?subtree_as_ids' % {'project_id': projects[0]['project']['id']})
    self.assertValidProjectResponse(r, projects[0]['project'])
    subtree_as_ids = r.result['project']['subtree']
    expected_dict = {projects[1]['project']['id']: {projects[2]['project']['id']: None, projects[4]['project']['id']: None}, projects[3]['project']['id']: None}
    self.assertDictEqual(expected_dict, subtree_as_ids)
    r = self.get('/projects/%(project_id)s?subtree_as_ids' % {'project_id': projects[1]['project']['id']})
    self.assertValidProjectResponse(r, projects[1]['project'])
    subtree_as_ids = r.result['project']['subtree']
    expected_dict = {projects[2]['project']['id']: None, projects[4]['project']['id']: None}
    self.assertDictEqual(expected_dict, subtree_as_ids)
    r = self.get('/projects/%(project_id)s?subtree_as_ids' % {'project_id': projects[3]['project']['id']})
    self.assertValidProjectResponse(r, projects[3]['project'])
    subtree_as_ids = r.result['project']['subtree']
    self.assertIsNone(subtree_as_ids)