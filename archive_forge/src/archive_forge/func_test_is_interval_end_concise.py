import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import (
from aniso8601.tests.compat import mock
def test_is_interval_end_concise(self):
    self.assertTrue(BaseTimeBuilder._is_interval_end_concise(TimeTuple('1', '2', '3', None)))
    self.assertTrue(BaseTimeBuilder._is_interval_end_concise(DateTuple(None, '2', '3', '4', '5', '6')))
    self.assertTrue(BaseTimeBuilder._is_interval_end_concise(DatetimeTuple(DateTuple(None, '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', None))))
    self.assertFalse(BaseTimeBuilder._is_interval_end_concise(DateTuple('1', '2', '3', '4', '5', '6')))
    self.assertFalse(BaseTimeBuilder._is_interval_end_concise(DatetimeTuple(DateTuple('1', '2', '3', '4', '5', '6'), TimeTuple('7', '8', '9', None))))