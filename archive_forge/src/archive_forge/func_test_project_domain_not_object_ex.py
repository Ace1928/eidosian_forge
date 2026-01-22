from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_project_domain_not_object_ex(self):
    p = {'identity': {'methods': []}, 'scope': {'project': {'id': 'something', 'domain': 'something'}}}
    self._expect_failure(p)