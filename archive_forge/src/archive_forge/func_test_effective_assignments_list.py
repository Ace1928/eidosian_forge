from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import role_assignments
def test_effective_assignments_list(self):
    ref_list = self.TEST_USER_PROJECT_LIST + self.TEST_USER_DOMAIN_LIST
    self.stub_entity('GET', [self.collection_key, '?effective=True'], entity=ref_list)
    returned_list = self.manager.list(effective=True)
    self._assert_returned_list(ref_list, returned_list)
    kwargs = {'effective': 'True'}
    self.assertQueryStringContains(**kwargs)