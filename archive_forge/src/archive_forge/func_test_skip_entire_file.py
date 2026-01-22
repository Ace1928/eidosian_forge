from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_skip_entire_file(self):
    self.tap.write(_u('1..0 # Skipped: entire file skipped\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.check_events([('status', 'file skip', 'skip', None, True, 'tap comment', b'Skipped: entire file skipped', True, None, None, None)])