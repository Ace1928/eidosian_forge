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
def test_download_without_persistent_tracker(self):
    """
        Tests a single resumable download, with no tracker persistence
        """
    res_download_handler = ResumableDownloadHandler()
    dst_fp = self.make_dst_fp()
    small_src_key_as_string, small_src_key = self.make_small_key()
    small_src_key.get_contents_to_file(dst_fp, res_download_handler=res_download_handler)
    self.assertEqual(SMALL_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(small_src_key_as_string, small_src_key.get_contents_as_string())