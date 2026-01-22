from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import services
def test_services_disable_log_reason(self):
    s = cs.services.disable_log_reason('host1', 'cinder-volume', 'disable bad host')
    values = {'host': 'host1', 'binary': 'cinder-volume', 'disabled_reason': 'disable bad host'}
    cs.assert_called('PUT', '/os-services/disable-log-reason', values)
    self.assertIsInstance(s, services.Service)
    self.assertEqual('disabled', s.status)
    self._assert_request_id(s)