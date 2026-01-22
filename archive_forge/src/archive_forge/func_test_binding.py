import datetime
import uuid
from oslo_utils import timeutils
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_binding(self):
    token = fixture.V3Token()
    principal = uuid.uuid4().hex
    token.set_bind('kerberos', principal)
    auth_ref = access.create(body=token)
    self.assertIsInstance(auth_ref, access.AccessInfoV3)
    self.assertEqual({'kerberos': principal}, auth_ref.bind)