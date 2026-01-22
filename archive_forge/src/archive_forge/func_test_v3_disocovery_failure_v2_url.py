import uuid
from keystoneauth1.identity.generic import password
from keystoneauth1.identity import v2
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import password as v3_password
from keystoneauth1.tests.unit.identity import utils
def test_v3_disocovery_failure_v2_url(self):
    auth_url = self.TEST_URL + 'v2.0'
    self.stub_url('GET', json={}, base_url='/v2.0', status_code=500)
    self.assertDiscoveryFailure(domain_id=uuid.uuid4().hex, auth_url=auth_url)