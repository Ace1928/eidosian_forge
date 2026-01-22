from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_get_encryption_metadata(self):
    vol = cs.volumes.get_encryption_metadata('1234')
    cs.assert_called('GET', '/volumes/1234/encryption')
    self._assert_request_id(vol)