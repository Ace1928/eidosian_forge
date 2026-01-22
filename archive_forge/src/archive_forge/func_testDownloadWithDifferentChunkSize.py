from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import os
import pkgutil
import six
import gslib.cloud_api
from gslib.daisy_chain_wrapper import DaisyChainWrapper
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
def testDownloadWithDifferentChunkSize(self):
    """Tests multiple calls to GetObjectMedia."""
    upload_file = self.CreateTempFile()
    write_values = []
    with open(self.test_data_file, 'rb') as stream:
        buffer_write_size = TRANSFER_BUFFER_SIZE * 2 + 10
        while True:
            data = stream.read(buffer_write_size)
            if not data:
                break
            write_values.append(data)
    mock_api = self.MockDownloadCloudApi(write_values)
    daisy_chain_wrapper = DaisyChainWrapper(self._dummy_url, self.test_data_file_len, mock_api, download_chunk_size=TRANSFER_BUFFER_SIZE)
    self._WriteFromWrapperToFile(daisy_chain_wrapper, upload_file)
    num_expected_calls = math.ceil(self.test_data_file_len / TRANSFER_BUFFER_SIZE)
    self.assertEqual(mock_api.get_calls, num_expected_calls)
    with open(upload_file, 'rb') as upload_stream:
        with open(self.test_data_file, 'rb') as download_stream:
            self.assertEqual(upload_stream.read(), download_stream.read())