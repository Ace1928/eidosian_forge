import unittest
import aniso8601
from aniso8601.exceptions import ISOFormatError
from aniso8601.tests.compat import mock
from aniso8601.timezone import parse_timezone
def test_parse_timezone_badtype(self):
    testtuples = (None, 1, False, 1.234)
    for testtuple in testtuples:
        with self.assertRaises(ValueError):
            parse_timezone(testtuple, builder=None)