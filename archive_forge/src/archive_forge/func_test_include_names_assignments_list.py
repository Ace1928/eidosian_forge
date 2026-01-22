from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_include_names_assignments_list(self):
    ref_list = self.TEST_ALL_RESPONSE_LIST
    self.stub_entity('GET', [self.collection_key, '?include_names=True'], entity=ref_list)
    returned_list = self.manager.list(include_names=True)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'include_names': 'True'}
    self.assertQueryStringContains(**kwargs)