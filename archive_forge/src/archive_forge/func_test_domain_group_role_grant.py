import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_domain_group_role_grant(self):
    group_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    ref = self.new_ref()
    self.stub_url('PUT', ['domains', domain_id, 'groups', group_id, self.collection_key, ref['id']], status_code=201)
    self.manager.grant(role=ref['id'], domain=domain_id, group=group_id)