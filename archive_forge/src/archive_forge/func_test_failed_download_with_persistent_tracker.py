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
def test_failed_download_with_persistent_tracker(self):
    """
        Tests that failed resumable download leaves a correct tracker file
        """
    harness = CallbackTestHarness()
    tmpdir = self._MakeTempDir()
    tracker_file_name = self.make_tracker_file(tmpdir)
    dst_fp = self.make_dst_fp(tmpdir)
    res_download_handler = ResumableDownloadHandler(tracker_file_name=tracker_file_name, num_retries=0)
    small_src_key_as_string, small_src_key = self.make_small_key()
    try:
        small_src_key.get_contents_to_file(dst_fp, cb=harness.call, res_download_handler=res_download_handler)
        self.fail('Did not get expected ResumableDownloadException')
    except ResumableDownloadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT_CUR_PROCESS)
        self.assertTrue(os.path.exists(tracker_file_name))
        f = open(tracker_file_name)
        etag_line = f.readline()
        self.assertEquals(etag_line.rstrip('\n'), small_src_key.etag.strip('"\''))