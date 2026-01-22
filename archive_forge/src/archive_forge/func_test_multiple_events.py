import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_multiple_events(self):
    source = BytesIO(CONSTANT_ENUM + CONSTANT_ENUM)
    result = StreamResult()
    subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout').run(result)
    self.assertEqual(b'', source.read())
    self.assertEqual([('status', 'foo', 'exists', None, True, None, None, False, None, None, None), ('status', 'foo', 'exists', None, True, None, None, False, None, None, None)], result._events)