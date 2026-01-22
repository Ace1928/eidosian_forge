import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import roles
from keystoneclient.v2_0 import users
def test_user_role_listing(self):
    user_id = uuid.uuid4().hex
    role_id1 = uuid.uuid4().hex
    role_id2 = uuid.uuid4().hex
    tenant_id = uuid.uuid4().hex
    user_resp = {'user': {'id': user_id, 'email': uuid.uuid4().hex, 'name': uuid.uuid4().hex}}
    roles_resp = {'roles': {'values': [{'name': uuid.uuid4().hex, 'id': role_id1}, {'name': uuid.uuid4().hex, 'id': role_id2}]}}
    self.stub_url('GET', ['users', user_id], json=user_resp)
    self.stub_url('GET', ['tenants', tenant_id, 'users', user_id, 'roles'], json=roles_resp)
    user = self.client.users.get(user_id)
    role_objs = user.list_roles(tenant_id)
    for r in role_objs:
        self.assertIsInstance(r, roles.Role)
    self.assertEqual(set([role_id1, role_id2]), set([r.id for r in role_objs]))