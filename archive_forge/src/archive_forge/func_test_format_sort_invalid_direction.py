from cinderclient import api_versions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3.volumes import Volume
def test_format_sort_invalid_direction(self):
    for s in ['id:foo', 'id:asc,status,size:foo', ['id', 'status', 'size:foo']]:
        self.assertRaises(ValueError, cs.volumes._format_sort_param, s)