import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
@given(st.binary())
def test_hypothesis_decoding(self, code_bytes):
    source = BytesIO(code_bytes)
    result = StreamResult()
    stream = subunit.ByteStreamToStreamResult(source, non_subunit_name='stdout')
    stream.run(result)
    self.assertEqual(b'', source.read())