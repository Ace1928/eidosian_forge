from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_create_snap(self):
    vs = self.cs.assisted_volume_snapshots.create('1', {})
    self.assert_request_id(vs, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('POST', '/os-assisted-volume-snapshots')