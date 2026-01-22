from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_bail_out_errors(self):
    self.tap.write(_u('ok 1 foo\n'))
    self.tap.write(_u('Bail out! Lifejacket engaged\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.check_events([('status', 'test 1 foo', 'success', None, False, None, None, True, None, None, None), ('status', 'Bail out! Lifejacket engaged', 'fail', None, False, None, None, True, None, None, None)])