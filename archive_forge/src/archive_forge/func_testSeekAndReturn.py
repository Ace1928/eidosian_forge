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
def testSeekAndReturn(self):
    """Tests seeking to the end of the wrapper (simulates getting size)."""
    write_values = []
    with open(self.test_data_file, 'rb') as stream:
        while True:
            data = stream.read(TRANSFER_BUFFER_SIZE)
            if not data:
                break
            write_values.append(data)
    upload_file = self.CreateTempFile()
    mock_api = self.MockDownloadCloudApi(write_values)
    daisy_chain_wrapper = DaisyChainWrapper(self._dummy_url, self.test_data_file_len, mock_api, download_chunk_size=self.test_data_file_len)
    with open(upload_file, 'wb') as upload_stream:
        current_position = 0
        daisy_chain_wrapper.seek(0, whence=os.SEEK_END)
        daisy_chain_wrapper.seek(current_position)
        while True:
            data = daisy_chain_wrapper.read(TRANSFER_BUFFER_SIZE)
            current_position += len(data)
            daisy_chain_wrapper.seek(0, whence=os.SEEK_END)
            daisy_chain_wrapper.seek(current_position)
            if not data:
                break
            upload_stream.write(data)
    self.assertEqual(mock_api.get_calls, 1)
    with open(upload_file, 'rb') as upload_stream:
        with open(self.test_data_file, 'rb') as download_stream:
            self.assertEqual(upload_stream.read(), download_stream.read())