from datetime import (
import time
import unittest
def test_millisecond(self):
    x = datetime(2018, 9, 20, 13, 11, 21, 123000)
    self.assertEqual(format_millisecond(datetime(2018, 9, 20, 13, 11, 21, 123000), utc=True, use_system_timezone=False), '2018-09-20T13:11:21.123Z')