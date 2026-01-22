from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_types
def test_unset_multiple_keys(self):
    t = cs.volume_types.get(1)
    res = t.unset_keys(['k', 'm'])
    cs.assert_called_anytime('DELETE', '/types/1/extra_specs/k')
    cs.assert_called_anytime('DELETE', '/types/1/extra_specs/m')
    self._assert_request_id(res, count=2)