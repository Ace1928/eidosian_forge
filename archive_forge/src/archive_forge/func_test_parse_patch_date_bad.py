import random
import time
from breezy import tests, timestamp
from breezy.osutils import local_time_offset
def test_parse_patch_date_bad(self):
    self.assertRaises(ValueError, timestamp.parse_patch_date, 'NOT A TIME')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 -0500x')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03 10:04:19 -0500')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04 -0500')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 0500')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 +2400')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 -2400')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 -0560')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 79500')
    self.assertRaises(ValueError, timestamp.parse_patch_date, '2007-03-06 10:04:19 +05-5')