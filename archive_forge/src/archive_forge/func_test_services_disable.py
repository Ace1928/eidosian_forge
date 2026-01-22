from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import services
def test_services_disable(self):
    s = cs.services.disable('host1', 'cinder-volume')
    values = {'host': 'host1', 'binary': 'cinder-volume'}
    cs.assert_called('PUT', '/os-services/disable', values)
    self.assertIsInstance(s, services.Service)
    self.assertEqual('disabled', s.status)
    self._assert_request_id(s)