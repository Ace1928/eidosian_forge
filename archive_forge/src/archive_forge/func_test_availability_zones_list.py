import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import availability_zones as az
def test_availability_zones_list(self):
    zones = self.mgr.list()
    expect = [('GET', '/v1/availability_zones', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(zones, matchers.HasLength(2))