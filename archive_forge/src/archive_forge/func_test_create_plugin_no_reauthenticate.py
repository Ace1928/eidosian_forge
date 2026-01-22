import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_create_plugin_no_reauthenticate(self):
    self.stub_discovery()
    self.assertCreateV2(reauthenticate=False)
    self.assertCreateV3(domain_id=uuid.uuid4().hex, reauthenticate=False)