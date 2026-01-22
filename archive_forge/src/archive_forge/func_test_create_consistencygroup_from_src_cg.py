from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_consistencygroup_from_src_cg(self):
    vol = cs.consistencygroups.create_from_src(None, '5678', name='cg')
    expected = {'consistencygroup-from-src': {'status': 'creating', 'description': None, 'user_id': None, 'name': 'cg', 'source_cgid': '5678', 'project_id': None, 'cgsnapshot_id': None}}
    cs.assert_called('POST', '/consistencygroups/create_from_src', body=expected)
    self._assert_request_id(vol)