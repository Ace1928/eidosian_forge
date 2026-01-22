import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def test_domain_role_revoke(self):
    user_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    ref = self.new_ref()
    self.stub_url('DELETE', ['domains', domain_id, 'users', user_id, self.collection_key, ref['id']], status_code=204)
    self.manager.revoke(role=ref['id'], domain=domain_id, user=user_id)