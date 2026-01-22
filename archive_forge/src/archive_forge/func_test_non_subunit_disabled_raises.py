import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_non_subunit_disabled_raises(self):
    source = BytesIO(b'foo\nbar\n')
    result = StreamResult()
    case = subunit.ByteStreamToStreamResult(source)
    e = self.assertRaises(Exception, case.run, result)
    self.assertEqual(b'f', e.args[1])
    self.assertEqual(b'oo\nbar\n', source.read())
    self.assertEqual([], result._events)