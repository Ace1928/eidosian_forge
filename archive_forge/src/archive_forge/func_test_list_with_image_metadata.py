from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_list_with_image_metadata(self):
    cs = fakes.FakeClient(api_versions.APIVersion('3.0'))
    cs.volumes.list(search_opts={'glance_metadata': {'key1': 'val1'}})
    expected = '/volumes/detail?glance_metadata=%s' % parse.quote_plus("{'key1': 'val1'}")
    cs.assert_called('GET', expected)