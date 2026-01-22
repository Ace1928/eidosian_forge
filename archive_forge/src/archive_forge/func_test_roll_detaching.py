from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_roll_detaching(self):
    v = cs.volumes.get('1234')
    self._assert_request_id(v)
    vol = cs.volumes.roll_detaching(v)
    cs.assert_called('POST', '/volumes/1234/action')
    self._assert_request_id(vol)