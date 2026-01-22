import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_bad_crc_errors_via_status(self):
    file_bytes = CONSTANT_MIME[:-1] + b'\x00'
    self.check_events(file_bytes, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=file_bytes, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'Bad checksum - calculated (0x78335115), stored (0x78335100)', mime_type='text/plain;charset=utf8')])