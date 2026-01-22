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
def test_discovering_version_with_discovery(self):
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    v2_compute = self.TEST_COMPUTE_ADMIN + '/v2.0'
    v3_compute = self.TEST_COMPUTE_ADMIN + '/v3'
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_v2(v2_compute)
    disc.add_v3(v3_compute)
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
    version = s.get_api_major_version(service_type='compute', interface='admin')
    self.assertEqual((3, 0), version)
    self.assertEqual(self.requests_mock.request_history[-1].url, self.TEST_COMPUTE_ADMIN)