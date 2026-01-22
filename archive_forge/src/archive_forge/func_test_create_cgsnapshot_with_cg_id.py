from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_create_cgsnapshot_with_cg_id(self):
    vol = cs.cgsnapshots.create('1234')
    expected = {'cgsnapshot': {'status': 'creating', 'description': None, 'user_id': None, 'name': None, 'consistencygroup_id': '1234', 'project_id': None}}
    cs.assert_called('POST', '/cgsnapshots', body=expected)
    self._assert_request_id(vol)