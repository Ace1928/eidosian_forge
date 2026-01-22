from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_delete_metadata(self):
    keys = ['key1']
    vol = cs.volumes.delete_metadata(1234, keys)
    cs.assert_called('DELETE', '/volumes/1234/metadata/key1')
    self._assert_request_id(vol)