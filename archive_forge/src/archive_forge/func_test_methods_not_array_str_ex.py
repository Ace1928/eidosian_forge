from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_methods_not_array_str_ex(self):
    p = {'identity': {'methods': [{}]}}
    self._expect_failure(p)