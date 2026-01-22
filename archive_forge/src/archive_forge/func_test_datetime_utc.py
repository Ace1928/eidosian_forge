from datetime import (
import time
import unittest
def test_datetime_utc(self):
    d = datetime.now()
    d_utc = d - self.local_utcoffset
    self.assertEqual(rfc3339(d, utc=True), d_utc.strftime('%Y-%m-%dT%H:%M:%SZ'))