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
def testDownloadThreadException(self):
    """Tests that an exception is propagated via the upload thread."""

    class DownloadException(Exception):
        pass
    write_values = [b'a', b'b', DownloadException('Download thread forces failure')]
    upload_file = self.CreateTempFile()
    mock_api = self.MockDownloadCloudApi(write_values)
    daisy_chain_wrapper = DaisyChainWrapper(self._dummy_url, self.test_data_file_len, mock_api, download_chunk_size=self.test_data_file_len)
    try:
        self._WriteFromWrapperToFile(daisy_chain_wrapper, upload_file)
        self.fail('Expected exception')
    except DownloadException as e:
        self.assertIn('Download thread forces failure', str(e))