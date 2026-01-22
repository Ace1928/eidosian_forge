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
def testReadSeekAndReadToEOF(self):
    """Tests performing reads on the wrapper, seeking, then reading to EOF."""
    for initial_reads in ([1], [TRANSFER_BUFFER_SIZE - 1], [TRANSFER_BUFFER_SIZE], [TRANSFER_BUFFER_SIZE + 1], [1, TRANSFER_BUFFER_SIZE - 1], [1, TRANSFER_BUFFER_SIZE], [1, TRANSFER_BUFFER_SIZE + 1], [TRANSFER_BUFFER_SIZE - 1, 1], [TRANSFER_BUFFER_SIZE, 1], [TRANSFER_BUFFER_SIZE + 1, 1], [TRANSFER_BUFFER_SIZE - 1, TRANSFER_BUFFER_SIZE - 1], [TRANSFER_BUFFER_SIZE - 1, TRANSFER_BUFFER_SIZE], [TRANSFER_BUFFER_SIZE - 1, TRANSFER_BUFFER_SIZE + 1], [TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE - 1], [TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE], [TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE + 1], [TRANSFER_BUFFER_SIZE + 1, TRANSFER_BUFFER_SIZE - 1], [TRANSFER_BUFFER_SIZE + 1, TRANSFER_BUFFER_SIZE], [TRANSFER_BUFFER_SIZE + 1, TRANSFER_BUFFER_SIZE + 1], [TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE]):
        initial_position = 0
        for read_size in initial_reads:
            initial_position += read_size
        for buffer_size in (initial_position, initial_position + 1, initial_position * 2 - 1, initial_position * 2):
            for seek_back_amount in (min(TRANSFER_BUFFER_SIZE - 1, initial_position), min(TRANSFER_BUFFER_SIZE, initial_position), min(TRANSFER_BUFFER_SIZE + 1, initial_position), min(TRANSFER_BUFFER_SIZE * 2 - 1, initial_position), min(TRANSFER_BUFFER_SIZE * 2, initial_position), min(TRANSFER_BUFFER_SIZE * 2 + 1, initial_position)):
                self._testSeekBack(initial_reads, buffer_size, seek_back_amount)