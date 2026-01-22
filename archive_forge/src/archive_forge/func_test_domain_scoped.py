import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_domain_scoped(self):
    domain_id = uuid.uuid4().hex
    domain_name = uuid.uuid4().hex
    token = fixture.V3Token(domain_id=domain_id, domain_name=domain_name)
    self.assertEqual(domain_id, token.domain_id)
    self.assertEqual(domain_id, token['token']['domain']['id'])
    self.assertEqual(domain_name, token.domain_name)
    self.assertEqual(domain_name, token['token']['domain']['name'])