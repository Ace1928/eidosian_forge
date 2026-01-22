from unittest import TestCase
from fastimport import (
def test_parse_tz_cet(self):
    self.assertEqual(3600, dates.parse_tz(b'+0100'))