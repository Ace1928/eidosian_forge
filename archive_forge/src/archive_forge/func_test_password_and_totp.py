import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_password_and_totp(self):
    username = uuid.uuid4().hex
    password = uuid.uuid4().hex
    user_domain_id = uuid.uuid4().hex
    passcode = ''.join((str(random.randint(0, 9)) for x in range(6)))
    project_name = uuid.uuid4().hex
    project_domain_id = uuid.uuid4().hex
    p = self.create(auth_methods=['v3password', 'v3totp'], username=username, password=password, user_domain_id=user_domain_id, project_name=project_name, project_domain_id=project_domain_id, passcode=passcode)
    password_method = p.auth_methods[0]
    totp_method = p.auth_methods[1]
    self.assertEqual(username, password_method.username)
    self.assertEqual(user_domain_id, password_method.user_domain_id)
    self.assertEqual(password, password_method.password)
    self.assertEqual(username, totp_method.username)
    self.assertEqual(user_domain_id, totp_method.user_domain_id)
    self.assertEqual(passcode, totp_method.passcode)
    self.assertEqual(project_name, p.project_name)
    self.assertEqual(project_domain_id, p.project_domain_id)