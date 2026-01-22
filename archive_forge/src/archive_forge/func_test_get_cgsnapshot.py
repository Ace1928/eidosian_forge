from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_get_cgsnapshot(self):
    cgsnapshot_id = '1234'
    vol = cs.cgsnapshots.get(cgsnapshot_id)
    cs.assert_called('GET', '/cgsnapshots/%s' % cgsnapshot_id)
    self._assert_request_id(vol)