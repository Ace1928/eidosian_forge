import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_get_with_invalid_parameters_combination(self):
    self.assertRaises(ksc_exceptions.ValidationError, self.manager.get, project=uuid.uuid4().hex, subtree_as_list=True, subtree_as_ids=True)
    self.assertRaises(ksc_exceptions.ValidationError, self.manager.get, project=uuid.uuid4().hex, parents_as_list=True, parents_as_ids=True)