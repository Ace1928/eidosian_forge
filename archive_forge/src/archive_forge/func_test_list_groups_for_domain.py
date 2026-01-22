import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import groups
def test_list_groups_for_domain(self):
    ref_list = [self.new_ref(), self.new_ref()]
    domain_id = uuid.uuid4().hex
    self.stub_entity('GET', [self.collection_key], status_code=200, entity=ref_list)
    returned_list = self.manager.list(domain=domain_id)
    self.assertTrue(len(ref_list), len(returned_list))
    [self.assertIsInstance(r, self.model) for r in returned_list]
    self.assertQueryStringIs('domain_id=%s' % domain_id)