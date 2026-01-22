import unittest
import aniso8601
from aniso8601.exceptions import ISOFormatError
from aniso8601.tests.compat import mock
from aniso8601.timezone import parse_timezone
def test_parse_timezone_badstr(self):
    testtuples = ('+1', '-00', '-0000', '-00:00', '01', '0123', '@12:34', 'Y', ' Z', 'Z ', ' Z ', 'bad', '')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            parse_timezone(testtuple, builder=None)