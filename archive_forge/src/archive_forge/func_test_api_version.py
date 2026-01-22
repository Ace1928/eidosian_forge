from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import services
def test_api_version(self):
    client = fakes.FakeClient(version_header='3.0')
    svs = client.services.server_api_version()
    [self.assertIsInstance(s, services.Service) for s in svs]