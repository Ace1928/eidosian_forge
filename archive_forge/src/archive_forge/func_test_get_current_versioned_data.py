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
def test_get_current_versioned_data(self):
    v2_compute = self.TEST_COMPUTE_ADMIN + '/v2.0'
    v3_compute = self.TEST_COMPUTE_ADMIN + '/v3'
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_v2(v2_compute)
    disc.add_v3(v3_compute)
    resps = [{'json': disc}, {'status_code': 500}]
    self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    data = a.get_endpoint_data(session=s, service_type='compute', interface='admin')
    self.assertEqual(v3_compute, data.url)
    v3_data = data.get_current_versioned_data(s)
    self.assertEqual(v3_compute, v3_data.url)
    self.assertEqual(v3_compute, v3_data.service_url)
    self.assertEqual(self.TEST_COMPUTE_ADMIN, v3_data.catalog_url)
    self.assertEqual((3, 0), v3_data.api_version)
    self.assertIsNone(v3_data.min_microversion)
    self.assertIsNone(v3_data.max_microversion)