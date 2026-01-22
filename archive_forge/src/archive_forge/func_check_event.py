import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def check_event(self, source_bytes, test_status=None, test_id='foo', route_code=None, timestamp=None, tags=None, mime_type=None, file_name=None, file_bytes=None, eof=False, runnable=True):
    event = self._event(test_id=test_id, test_status=test_status, tags=tags, runnable=runnable, file_name=file_name, file_bytes=file_bytes, eof=eof, mime_type=mime_type, route_code=route_code, timestamp=timestamp)
    self.check_events(source_bytes, [event])