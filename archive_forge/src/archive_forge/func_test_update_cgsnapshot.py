from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_cgsnapshot(self):
    v = cs.cgsnapshots.list()[0]
    expected = {'cgsnapshot': {'name': 'cgs2'}}
    vol = v.update(name='cgs2')
    cs.assert_called('PUT', '/cgsnapshots/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.cgsnapshots.update('1234', name='cgs2')
    cs.assert_called('PUT', '/cgsnapshots/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.cgsnapshots.update(v, name='cgs2')
    cs.assert_called('PUT', '/cgsnapshots/1234', body=expected)
    self._assert_request_id(vol)