from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_password_user_domain_no_id_or_name_ex(self):
    p = {'identity': {'methods': ['password'], 'password': {'user': {'id': 'something', 'domain': {}}}}}
    self._expect_failure(p)