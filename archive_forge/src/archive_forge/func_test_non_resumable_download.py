import errno
import os
import re
import boto
from boto.s3.resumable_download_handler import get_cur_file_size
from boto.s3.resumable_download_handler import ResumableDownloadHandler
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableDownloadException
from .cb_test_harness import CallbackTestHarness
from tests.integration.gs.testcase import GSTestCase
def test_non_resumable_download(self):
    """
        Tests that non-resumable downloads work
        """
    dst_fp = self.make_dst_fp()
    small_src_key_as_string, small_src_key = self.make_small_key()
    small_src_key.get_contents_to_file(dst_fp)
    self.assertEqual(SMALL_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(small_src_key_as_string, small_src_key.get_contents_as_string())