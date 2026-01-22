from datetime import (
import time
import unittest
def testPSTChange(self):
    """Test Standard time change"""
    self.assertEqual(rfc3339(datetime(2010, 11, 7, 0, 59)), '2010-11-07T00:59:00-07:00')
    self.assertEqual(rfc3339(datetime(2010, 11, 7, 1, 0)), '2010-11-07T01:00:00-07:00')