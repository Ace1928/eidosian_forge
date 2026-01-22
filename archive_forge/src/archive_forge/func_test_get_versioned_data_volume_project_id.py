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
def test_get_versioned_data_volume_project_id(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v3'].discovery.public, id='v3.0', status='CURRENT', min_version='3.0', version='3.20')
    disc.add_nova_microversion(href=self.TEST_VOLUME.versions['v2'].discovery.public, id='v2.0', status='SUPPORTED')
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    endpoint = a.get_endpoint(session=s, service_type='volumev3', interface='public', version='3.0')
    self.assertEqual(self.TEST_VOLUME.catalog.public, endpoint)
    resps = [{'json': disc}, {'status_code': 500}]
    self.requests_mock.get(self.TEST_VOLUME.versions['v3'].discovery.public + '/', resps)
    data = a.get_endpoint_data(session=s, service_type='volumev3', interface='public')
    self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, data.url)
    v3_data = data.get_versioned_data(s, min_version='3.0', max_version='3.latest', project_id=self.project_id)
    self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, v3_data.url)
    self.assertEqual(self.TEST_VOLUME.catalog.public, v3_data.catalog_url)
    self.assertEqual((3, 0), v3_data.min_microversion)
    self.assertEqual((3, 20), v3_data.max_microversion)
    self.assertEqual(self.TEST_VOLUME.versions['v3'].service.public, v3_data.service_url)
    self.requests_mock.get(self.TEST_VOLUME.unversioned.public, resps)
    v2_data = data.get_versioned_data(s, min_version='2.0', max_version='2.latest', project_id=self.project_id)
    self.assertEqual(self.TEST_VOLUME.versions['v2'].service.public, v2_data.url)
    self.assertEqual(self.TEST_VOLUME.versions['v2'].service.public, v2_data.service_url)
    self.assertEqual(self.TEST_VOLUME.catalog.public, v2_data.catalog_url)
    self.assertIsNone(v2_data.min_microversion)
    self.assertIsNone(v2_data.max_microversion)