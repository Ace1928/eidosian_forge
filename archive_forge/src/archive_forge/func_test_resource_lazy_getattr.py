import uuid
import fixtures
from keystoneauth1.identity import v2
from keystoneauth1 import session
import requests
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.tests.unit import utils
from keystoneclient import utils as base_utils
from keystoneclient.v2_0 import client
from keystoneclient.v2_0 import roles
from keystoneclient.v3 import users
def test_resource_lazy_getattr(self):
    auth = v2.Token(token=self.TEST_TOKEN, auth_url='http://127.0.0.1:5000')
    session_ = session.Session(auth=auth)
    self.client = client.Client(session=session_)
    self.useFixture(fixtures.MockPatchObject(self.client._adapter, 'get', side_effect=AttributeError, autospec=True))
    f = roles.Role(self.client.roles, {'id': 1, 'name': 'Member'})
    self.assertEqual(f.name, 'Member')
    self.assertRaises(AttributeError, getattr, f, 'blahblah')