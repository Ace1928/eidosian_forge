from unittest import TestCase
from fastimport import (
def test_parse_tz_odd(self):
    self.assertEqual(1864800, dates.parse_tz(b'+51800'))