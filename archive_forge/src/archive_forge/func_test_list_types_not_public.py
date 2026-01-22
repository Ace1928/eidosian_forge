from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_types
def test_list_types_not_public(self):
    t1 = cs.volume_types.list(is_public=None)
    cs.assert_called('GET', '/types?is_public=None')
    self._assert_request_id(t1)