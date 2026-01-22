from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_volume_list_manageable(self):
    cs.volumes.list_manageable('host1', detailed=False)
    cs.assert_called('GET', '/os-volume-manage?host=host1')