from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_volume_manage(self):
    vol = cs.volumes.manage('host1', {'k': 'v'})
    expected = {'host': 'host1', 'name': None, 'availability_zone': None, 'description': None, 'metadata': None, 'ref': {'k': 'v'}, 'volume_type': None, 'bootable': False}
    cs.assert_called('POST', '/os-volume-manage', {'volume': expected})
    self._assert_request_id(vol)