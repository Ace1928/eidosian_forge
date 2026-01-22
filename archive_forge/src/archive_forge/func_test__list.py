from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test__list(self):
    limit = 2
    url = '/volumes?limit=%s' % limit
    response_key = 'volumes'
    fake_volume1234 = Volume(self, {'id': 1234, 'name': 'sample-volume'}, loaded=True)
    fake_volume5678 = Volume(self, {'id': 5678, 'name': 'sample-volume2'}, loaded=True)
    fake_volumes = [fake_volume1234, fake_volume5678]
    volumes = cs.volumes._list(url, response_key, limit=limit)
    self._assert_request_id(volumes)
    cs.assert_called('GET', url)
    self.assertEqual(fake_volumes, volumes)
    cs.client.osapi_max_limit = 1
    volumes = cs.volumes._list(url, response_key, limit=limit)
    self.assertEqual(fake_volumes, volumes)
    self._assert_request_id(volumes)
    cs.client.osapi_max_limit = 1000