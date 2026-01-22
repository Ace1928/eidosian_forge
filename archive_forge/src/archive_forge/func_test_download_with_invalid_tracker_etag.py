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
def test_download_with_invalid_tracker_etag(self):
    """
        Tests resumable download with a tracker file containing an invalid etag
        """
    tmp_dir = self._MakeTempDir()
    dst_fp = self.make_dst_fp(tmp_dir)
    small_src_key_as_string, small_src_key = self.make_small_key()
    invalid_etag_tracker_file_name = os.path.join(tmp_dir, 'invalid_etag_tracker')
    f = open(invalid_etag_tracker_file_name, 'w')
    f.write('3.14159\n')
    f.close()
    res_download_handler = ResumableDownloadHandler(tracker_file_name=invalid_etag_tracker_file_name)
    small_src_key.get_contents_to_file(dst_fp, res_download_handler=res_download_handler)
    self.assertEqual(SMALL_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(small_src_key_as_string, small_src_key.get_contents_as_string())