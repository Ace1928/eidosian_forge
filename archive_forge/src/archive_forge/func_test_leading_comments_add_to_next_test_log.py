from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_leading_comments_add_to_next_test_log(self):
    self.tap.write(_u('# comment\n'))
    self.tap.write(_u('ok\n'))
    self.tap.write(_u('ok\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.check_events([('status', 'test 1', 'success', None, False, 'tap comment', b'# comment', True, 'text/plain; charset=UTF8', None, None), ('status', 'test 2', 'success', None, False, None, None, True, None, None, None)])