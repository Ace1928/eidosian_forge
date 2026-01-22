import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_route_code_and_file_content(self):
    content = BytesIO()
    subunit.StreamResultToBytes(content).status(route_code='0', mime_type='text/plain', file_name='bar', file_bytes=b'foo')
    self.check_event(content.getvalue(), test_id=None, file_name='bar', route_code='0', mime_type='text/plain', file_bytes=b'foo')