import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_is_admin_project_unset(self):
    token = fixture.V3Token()
    auth_ref = access.create(body=token)
    self.assertIsInstance(auth_ref, access.AccessInfoV3)
    self.assertIs(True, auth_ref.is_admin_project)