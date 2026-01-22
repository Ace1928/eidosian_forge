from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
def test_todo_and_skip(self):
    self.tap.write(_u('not ok 1 - a fail but # TODO but is TODO\n'))
    self.tap.write(_u('not ok 2 - another fail # SKIP instead\n'))
    self.tap.seek(0)
    result = subunit.TAP2SubUnit(self.tap, self.subunit)
    self.assertEqual(0, result)
    self.subunit.seek(0)
    events = StreamResult()
    subunit.ByteStreamToStreamResult(self.subunit).run(events)
    self.check_events([('status', 'test 1 - a fail but', 'xfail', None, False, 'tap comment', b'but is TODO', True, 'text/plain; charset=UTF8', None, None), ('status', 'test 2 - another fail', 'skip', None, False, 'tap comment', b'instead', True, 'text/plain; charset=UTF8', None, None)])