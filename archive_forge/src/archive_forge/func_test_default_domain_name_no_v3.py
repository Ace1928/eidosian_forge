import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_default_domain_name_no_v3(self):
    self.stub_discovery(v3=False)
    project_name = uuid.uuid4().hex
    default_domain_name = uuid.uuid4().hex
    p = self.assertCreateV2(project_name=project_name, default_domain_name=default_domain_name)
    self.assertEqual(project_name, p._plugin.tenant_name)