from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@ddt.data((False, '/volumes/summary'), (True, '/volumes/summary?all_tenants=True'))
def test_volume_summary(self, all_tenants_input):
    all_tenants, url = all_tenants_input
    cs = fakes.FakeClient(api_versions.APIVersion('3.12'))
    cs.volumes.summary(all_tenants=all_tenants)
    cs.assert_called('GET', url)