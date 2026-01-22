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
def test_broken_discovery_endpoint(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_nova_microversion(href='http://internal.example.com', id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    self.requests_mock.get(self.TEST_COMPUTE_PUBLIC, json=disc)
    data = s.get_endpoint_data(service_type='compute', interface='public', min_version='2.1', max_version='2.latest')
    self.assertTrue(data.url, self.TEST_COMPUTE_PUBLIC + '/v2.1')