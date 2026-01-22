import uuid
from keystoneauth1.identity.generic import token
from keystoneauth1.identity import v2
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import token as v3_token
from keystoneauth1.tests.unit.identity import utils
def test_token_cache_id(self):
    the_token = uuid.uuid4().hex
    project_name = uuid.uuid4().hex
    default_domain_id = uuid.uuid4().hex
    a = token.Token(self.TEST_URL, token=the_token, project_name=project_name, default_domain_id=default_domain_id)
    b = token.Token(self.TEST_URL, token=the_token, project_name=project_name, default_domain_id=default_domain_id)
    a_id = a.get_cache_id()
    b_id = b.get_cache_id()
    self.assertEqual(a_id, b_id)
    c = token.Token(self.TEST_URL, token=the_token, project_name=uuid.uuid4().hex, default_domain_id=default_domain_id)
    c_id = c.get_cache_id()
    self.assertNotEqual(a_id, c_id)