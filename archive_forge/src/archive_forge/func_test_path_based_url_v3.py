import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_path_based_url_v3(self):
    self.stub_url('GET', ['v3'], status_code=403)
    self.assertCreateV3(auth_url=self.TEST_URL + 'v3')