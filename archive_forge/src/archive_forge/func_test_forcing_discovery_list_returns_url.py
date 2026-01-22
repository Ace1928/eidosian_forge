import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_forcing_discovery_list_returns_url(self):
    common_disc = fixture.DiscoveryList(href=self.BASE_URL)
    v2_m = self.stub_url('GET', ['v2.0'], base_url=self.BASE_URL, status_code=200, json=common_disc)
    token = fixture.V2Token()
    service = token.add_service(self.IDENTITY)
    service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
    self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
    v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
    sess = session.Session(auth=v2_auth)
    self.assertFalse(v2_m.called)
    data = sess.get_endpoint_data(service_type=self.IDENTITY, discover_versions=True)
    self.assertTrue(v2_m.called)
    self.assertEqual(self.V2_URL, data.url)
    self.assertEqual((2, 0), data.api_version)