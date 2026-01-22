from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_password_user_password_not_string_ex(self):
    p = {'identity': {'methods': ['password'], 'password': {'user': {'id': 'something', 'password': {}}}}}
    self._expect_failure(p)