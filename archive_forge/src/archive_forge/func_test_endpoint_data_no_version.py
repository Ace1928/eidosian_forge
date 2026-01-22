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
def test_endpoint_data_no_version(self):
    path = self.TEST_COMPUTE_ADMIN[self.TEST_COMPUTE_ADMIN.find(':') + 1:]
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_v2(path + '/v2.0')
    disc.add_v3(path + '/v3')
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    data = a.get_endpoint_data(session=s, service_type='compute', interface='admin')
    self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', data.url)
    self.assertEqual((3, 0), data.api_version)