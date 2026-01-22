from keystone.auth import schema
from keystone import exception
from keystone.tests import unit
def test_project_domain_no_id_or_name_ex(self):
    p = {'identity': {'methods': []}, 'scope': {'project': {'id': 'something', 'domain': {}}}}
    self._expect_failure(p)