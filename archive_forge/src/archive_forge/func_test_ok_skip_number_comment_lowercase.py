from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_ok_skip_number_comment_lowercase(self):
    self.tap.write(_u('ok 1 # skip no samba environment available, skipping compilation\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.check_events([('status', 'test 1', 'skip', None, False, 'tap comment', b'no samba environment available, skipping compilation', True, 'text/plain; charset=UTF8', None, None)])