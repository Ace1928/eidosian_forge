import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_list_projects_for_parent(self):
    ref_list = [self.new_ref(), self.new_ref()]
    parent_id = uuid.uuid4().hex
    self.stub_entity('GET', [self.collection_key], entity=ref_list)
    returned_list = self.manager.list(parent=parent_id)
    self.assertEqual(len(ref_list), len(returned_list))
    [self.assertIsInstance(r, self.model) for r in returned_list]
    self.assertQueryStringIs('parent_id=%s' % parent_id)