import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_update_with_parent_project(self):
    ref = self.new_ref()
    ref['parent_id'] = uuid.uuid4().hex
    self.stub_entity('PATCH', id=ref['id'], entity=ref, status_code=403)
    req_ref = ref.copy()
    req_ref.pop('id')
    self.assertRaises(ksa_exceptions.Forbidden, self.manager.update, ref['id'], **utils.parameterize(req_ref))