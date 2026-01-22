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
def test_download_with_unwritable_tracker_file(self):
    """
        Tests resumable download with an unwritable tracker file
        """
    tmp_dir = self._MakeTempDir()
    tracker_file_name = os.path.join(tmp_dir, 'tracker')
    save_mod = os.stat(tmp_dir).st_mode
    try:
        os.chmod(tmp_dir, 0)
        res_download_handler = ResumableDownloadHandler(tracker_file_name=tracker_file_name)
    except ResumableDownloadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
        self.assertNotEqual(e.message.find("Couldn't write URI tracker file"), -1)
    finally:
        os.chmod(tmp_dir, save_mod)