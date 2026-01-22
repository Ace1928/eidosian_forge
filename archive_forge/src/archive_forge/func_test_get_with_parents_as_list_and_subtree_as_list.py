import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_get_with_parents_as_list_and_subtree_as_list(self):
    ref = self.new_ref()
    projects = self._create_projects_hierarchy()
    ref = projects[1]
    ref['parents_as_list'] = [projects[0]]
    ref['subtree_as_list'] = [projects[2]]
    self.stub_entity('GET', id=ref['id'], entity=ref)
    returned = self.manager.get(ref['id'], parents_as_list=True, subtree_as_list=True)
    self.assertQueryStringIs('subtree_as_list&parents_as_list')
    for attr in projects[0]:
        parent = getattr(returned, 'parents_as_list')[0]
        self.assertEqual(parent[attr], projects[0][attr], 'Expected different %s' % attr)
    for attr in projects[2]:
        child = getattr(returned, 'subtree_as_list')[0]
        self.assertEqual(child[attr], projects[2][attr], 'Expected different %s' % attr)