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
def test_broken_pipe_recovery(self):
    """
        Tests handling of a Broken Pipe (which interacts with an httplib bug)
        """
    exception = IOError(errno.EPIPE, 'Broken pipe')
    harness = CallbackTestHarness(exception=exception)
    res_download_handler = ResumableDownloadHandler(num_retries=1)
    dst_fp = self.make_dst_fp()
    small_src_key_as_string, small_src_key = self.make_small_key()
    small_src_key.get_contents_to_file(dst_fp, cb=harness.call, res_download_handler=res_download_handler)
    self.assertEqual(SMALL_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(small_src_key_as_string, small_src_key.get_contents_as_string())