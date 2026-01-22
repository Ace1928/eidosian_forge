import fixtures
import http.client as http
from oslo_utils import units
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional.v2.test_images import get_enforcer_class
from glance.tests import utils as test_utils
def test_stores(self):
    self.start_server()
    stores = self.api_get('/v2/info/stores').json['stores']
    expected = {'stores': [{'id': 'store1', 'default': 'true'}, {'id': 'store2'}, {'id': 'store3'}]}
    self.assertEqual(expected['stores'], stores)
    stores = self.api_get('/v2/info/stores/detail').json['stores']
    expected = {'stores': [{'id': 'store1', 'default': 'true', 'type': 'file', 'weight': 0, 'properties': {'data_dir': self._store_dir('store1'), 'chunk_size': 65536, 'thin_provisioning': False}}, {'id': 'store2', 'type': 'file', 'weight': 0, 'properties': {'data_dir': self._store_dir('store2'), 'chunk_size': 65536, 'thin_provisioning': False}}, {'id': 'store3', 'type': 'file', 'weight': 0, 'properties': {'data_dir': self._store_dir('store3'), 'chunk_size': 65536, 'thin_provisioning': False}}]}
    self.assertEqual(expected['stores'], stores)
    response = self.api_get('/v2/info/stores/detail', headers={'X-Roles': 'member'})
    self.assertEqual(http.FORBIDDEN, response.status_code)