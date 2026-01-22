from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_volume_list_manageable_detailed(self):
    cs.volumes.list_manageable('host1', detailed=True)
    cs.assert_called('GET', '/os-volume-manage/detail?host=host1')