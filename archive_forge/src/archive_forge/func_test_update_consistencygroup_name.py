from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_consistencygroup_name(self):
    v = cs.consistencygroups.list()[0]
    expected = {'consistencygroup': {'name': 'cg2'}}
    vol = v.update(name='cg2')
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.consistencygroups.update('1234', name='cg2')
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.consistencygroups.update(v, name='cg2')
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)