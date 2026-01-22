import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_file_content(self):
    self.check_event(CONSTANT_FILE_CONTENT, test_id=None, file_name='barney', file_bytes=b'woo')