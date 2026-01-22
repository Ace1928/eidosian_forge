import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
def test_range_check_timezone(self):
    self.assertEqual(BaseTimeBuilder.range_check_timezone(), (None, None, None, None, ''))
    self.assertEqual(BaseTimeBuilder.range_check_timezone(rangedict={}), (None, None, None, None, ''))