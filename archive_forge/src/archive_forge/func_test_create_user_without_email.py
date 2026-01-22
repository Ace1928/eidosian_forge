import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import roles
from keystoneclient.v2_0 import users
def test_create_user_without_email(self):
    tenant_id = uuid.uuid4().hex
    req_body = {'user': {'name': 'gabriel', 'password': 'test', 'tenantId': tenant_id, 'enabled': True, 'email': None}}
    user_id = uuid.uuid4().hex
    resp_body = {'user': {'name': 'gabriel', 'enabled': True, 'tenantId': tenant_id, 'id': user_id, 'password': 'test'}}
    self.stub_url('POST', ['users'], json=resp_body)
    user = self.client.users.create(req_body['user']['name'], req_body['user']['password'], tenant_id=req_body['user']['tenantId'], enabled=req_body['user']['enabled'])
    self.assertIsInstance(user, users.User)
    self.assertEqual(user.id, user_id)
    self.assertEqual(user.name, 'gabriel')
    self.assertRequestBodyIs(json=req_body)