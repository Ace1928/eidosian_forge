from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import services
def test_list_services_with_hostname(self):
    svs = cs.services.list(host='host2')
    cs.assert_called('GET', '/os-services?host=host2')
    self.assertEqual(2, len(svs))
    [self.assertIsInstance(s, services.Service) for s in svs]
    [self.assertEqual('host2', s.host) for s in svs]
    self._assert_request_id(svs)