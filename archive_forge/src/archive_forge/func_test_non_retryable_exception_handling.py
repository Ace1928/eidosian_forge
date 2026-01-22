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
def test_non_retryable_exception_handling(self):
    """
        Tests resumable download that fails with a non-retryable exception
        """
    harness = CallbackTestHarness(exception=OSError(errno.EACCES, 'Permission denied'))
    res_download_handler = ResumableDownloadHandler(num_retries=1)
    dst_fp = self.make_dst_fp()
    small_src_key_as_string, small_src_key = self.make_small_key()
    try:
        small_src_key.get_contents_to_file(dst_fp, cb=harness.call, res_download_handler=res_download_handler)
        self.fail('Did not get expected OSError')
    except OSError as e:
        self.assertEqual(e.errno, 13)