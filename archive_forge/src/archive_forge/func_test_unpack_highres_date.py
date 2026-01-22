import random
import time
from breezy import tests, timestamp
from breezy.osutils import local_time_offset
def test_unpack_highres_date(self):
    self.assertEqual((1120153132.35085, -18000), timestamp.unpack_highres_date('Thu 2005-06-30 12:38:52.350850105 -0500'))
    self.assertEqual((1120153132.35085, 0), timestamp.unpack_highres_date('Thu 2005-06-30 17:38:52.350850105 +0000'))
    self.assertEqual((1120153132.35085, 7200), timestamp.unpack_highres_date('Thu 2005-06-30 19:38:52.350850105 +0200'))
    self.assertEqual((1152428738.867522, 19800), timestamp.unpack_highres_date('Sun 2006-07-09 12:35:38.867522001 +0530'))