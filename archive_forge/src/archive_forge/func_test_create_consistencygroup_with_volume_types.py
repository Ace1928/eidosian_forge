from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_consistencygroup_with_volume_types(self):
    vol = cs.consistencygroups.create('type1,type2', 'cg')
    expected = {'consistencygroup': {'status': 'creating', 'description': None, 'availability_zone': None, 'user_id': None, 'name': 'cg', 'volume_types': 'type1,type2', 'project_id': None}}
    cs.assert_called('POST', '/consistencygroups', body=expected)
    self._assert_request_id(vol)