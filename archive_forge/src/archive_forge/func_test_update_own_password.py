import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import roles
from keystoneclient.v2_0 import users
def test_update_own_password(self):
    old_password = uuid.uuid4().hex
    new_password = uuid.uuid4().hex
    req_body = {'user': {'password': new_password, 'original_password': old_password}}
    resp_body = {'access': {}}
    self.stub_url('PATCH', ['OS-KSCRUD', 'users', self.TEST_USER_ID], json=resp_body)
    self.client.users.update_own_password(old_password, new_password)
    self.assertRequestBodyIs(json=req_body)
    self.assertNotIn(old_password, self.logger.output)
    self.assertNotIn(new_password, self.logger.output)