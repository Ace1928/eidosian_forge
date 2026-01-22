import uuid
from keystoneauth1.identity.generic import password
from keystoneauth1.identity import v2
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import password as v3_password
from keystoneauth1.tests.unit.identity import utils
def test_with_user_domain_params(self):
    self.stub_discovery()
    self.assertCreateV3(domain_id=uuid.uuid4().hex, user_domain_id=uuid.uuid4().hex)