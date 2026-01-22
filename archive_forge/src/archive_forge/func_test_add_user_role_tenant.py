import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import roles
def test_add_user_role_tenant(self):
    id_ = uuid.uuid4().hex
    self.stub_url('PUT', ['tenants', id_, 'users', 'foo', 'roles', 'OS-KSADM', 'barrr'], status_code=204)
    self.client.roles.add_user_role('foo', 'barrr', id_)