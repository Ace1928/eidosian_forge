import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_file_content_length_into_checksum(self):
    bad_file_length_content = b'\xb3!@\x13\x06barney\x04woo\xdc\xe2\xdb5'
    self.check_events(bad_file_length_content, [self._event(test_id='subunit.parser', eof=True, file_name='Packet data', file_bytes=bad_file_length_content, mime_type='application/octet-stream'), self._event(test_id='subunit.parser', test_status='fail', eof=True, file_name='Parser Error', file_bytes=b'File content extends past end of packet: claimed 4 bytes, 3 available', mime_type='text/plain;charset=utf8')])