from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_metadata_update_all(self):
    vol = cs.volumes.update_all_metadata(1234, {'k1': 'v1'})
    cs.assert_called('PUT', '/volumes/1234/metadata', {'metadata': {'k1': 'v1'}})
    self._assert_request_id(vol)