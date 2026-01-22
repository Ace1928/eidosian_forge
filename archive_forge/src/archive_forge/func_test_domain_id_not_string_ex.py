from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_domain_id_not_string_ex(self):
    p = {'identity': {'methods': []}, 'scope': {'domain': {'id': {}}}}
    self._expect_failure(p)