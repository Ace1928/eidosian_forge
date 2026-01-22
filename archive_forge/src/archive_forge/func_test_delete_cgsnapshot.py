from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_delete_cgsnapshot(self):
    v = cs.cgsnapshots.list()[0]
    vol = v.delete()
    self._assert_request_id(vol)
    cs.assert_called('DELETE', '/cgsnapshots/1234')
    vol = cs.cgsnapshots.delete('1234')
    cs.assert_called('DELETE', '/cgsnapshots/1234')
    self._assert_request_id(vol)
    vol = cs.cgsnapshots.delete(v)
    cs.assert_called('DELETE', '/cgsnapshots/1234')
    self._assert_request_id(vol)