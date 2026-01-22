from testtools import matchers
from heat.common import timeutils as util
from heat.tests import common
def test_valid_durations(self):
    self.assertEqual(0, util.parse_isoduration('PT'))
    self.assertEqual(3600, util.parse_isoduration('PT1H'))
    self.assertEqual(120, util.parse_isoduration('PT2M'))
    self.assertEqual(3, util.parse_isoduration('PT3S'))
    self.assertEqual(3900, util.parse_isoduration('PT1H5M'))
    self.assertEqual(3605, util.parse_isoduration('PT1H5S'))
    self.assertEqual(303, util.parse_isoduration('PT5M3S'))
    self.assertEqual(3903, util.parse_isoduration('PT1H5M3S'))
    self.assertEqual(24 * 3600, util.parse_isoduration('PT24H'))