from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_types
def test_list_types(self):
    tl = cs.volume_types.list()
    cs.assert_called('GET', '/types?is_public=None')
    self._assert_request_id(tl)
    for t in tl:
        self.assertIsInstance(t, volume_types.VolumeType)