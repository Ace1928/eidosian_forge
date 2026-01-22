from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
from six.moves import range
from gslib.exception import CommandException
from gslib.resumable_streaming_upload import ResumableStreamingJsonUploadWrapper
import gslib.tests.testcase as testcase
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.hashing_helper import CalculateHashesFromContents
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
def testSeekPartialBuffer(self):
    """Tests seeking back partially within the buffer."""
    tmp_file = self._GetTestFile()
    read_size = TRANSFER_BUFFER_SIZE
    with open(tmp_file, 'rb') as stream:
        wrapper = ResumableStreamingJsonUploadWrapper(stream, TRANSFER_BUFFER_SIZE * 3, test_small_buffer=True)
        position = 0
        for _ in range(3):
            data = wrapper.read(read_size)
            self.assertEqual(self._temp_test_file_contents[position:position + read_size], data, 'Data from position %s to %s did not match file contents.' % (position, position + read_size))
            position += len(data)
        data = wrapper.read(read_size // 2)
        position = read_size // 2
        wrapper.seek(position)
        data = wrapper.read()
        self.assertEqual(self._temp_test_file_contents[-len(data):], data, 'Data from position %s to EOF did not match file contents.' % position)