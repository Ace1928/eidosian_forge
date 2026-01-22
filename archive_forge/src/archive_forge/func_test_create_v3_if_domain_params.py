import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_create_v3_if_domain_params(self):
    self.stub_discovery()
    self.assertCreateV3(domain_id=uuid.uuid4().hex)
    self.assertCreateV3(domain_name=uuid.uuid4().hex)
    self.assertCreateV3(project_name=uuid.uuid4().hex, project_domain_name=uuid.uuid4().hex)
    self.assertCreateV3(project_name=uuid.uuid4().hex, project_domain_id=uuid.uuid4().hex)