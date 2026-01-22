import uuid
from keystoneauth1.identity.generic import password
from keystoneauth1.identity import v2
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import password as v3_password
from keystoneauth1.tests.unit.identity import utils
def test_v3_user_params_v2_url(self):
    self.stub_discovery(v3=False)
    self.assertDiscoveryFailure(user_domain_id=uuid.uuid4().hex)