from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_update_consistencygroup_remove_volumes(self):
    v = cs.consistencygroups.list()[0]
    uuids = 'uuid3,uuid4'
    expected = {'consistencygroup': {'remove_volumes': uuids}}
    vol = v.update(remove_volumes=uuids)
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.consistencygroups.update('1234', remove_volumes=uuids)
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)
    vol = cs.consistencygroups.update(v, remove_volumes=uuids)
    cs.assert_called('PUT', '/consistencygroups/1234', body=expected)
    self._assert_request_id(vol)