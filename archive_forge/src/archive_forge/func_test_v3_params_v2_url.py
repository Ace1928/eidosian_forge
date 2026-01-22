import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_v3_params_v2_url(self):
    self.stub_discovery(v3=False)
    self.assertDiscoveryFailure(domain_name=uuid.uuid4().hex)