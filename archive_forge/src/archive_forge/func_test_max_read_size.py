import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_max_read_size(self):
    """Read data in blocks and verify that the reads are not larger than
           the maximum read size.
        """
    mock_read_file = FakeReadFile(self.test_data)
    range_file = response.RangeFile('test_max_read_size', mock_read_file)
    response_data = range_file.read(self.test_data_len)
    self.assertTrue(mock_read_file.get_max_read_size() > 0)
    self.assertEqual(mock_read_file.get_max_read_size(), response.RangeFile._max_read_size)
    self.assertEqual(mock_read_file.get_read_count(), 3)
    if response_data != self.test_data:
        message = 'Data not equal.  Expected %d bytes, received %d.'
        self.fail(message % (len(response_data), self.test_data_len))