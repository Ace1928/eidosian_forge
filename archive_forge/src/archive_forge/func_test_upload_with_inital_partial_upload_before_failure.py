import errno
import random
import os
import time
from six import StringIO
import boto
from boto import storage_uri
from boto.gs.resumable_upload_handler import ResumableUploadHandler
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from .cb_test_harness import CallbackTestHarness
from tests.integration.gs.testcase import GSTestCase
def test_upload_with_inital_partial_upload_before_failure(self):
    """
        Tests resumable upload that successfully uploads some content
        before it fails, then restarts and completes
        """
    harness = CallbackTestHarness(fail_after_n_bytes=LARGE_KEY_SIZE / 2)
    res_upload_handler = ResumableUploadHandler(num_retries=1)
    larger_src_file_as_string, larger_src_file = self.make_large_file()
    larger_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    dst_key.set_contents_from_file(larger_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
    self.assertEqual(LARGE_KEY_SIZE, dst_key.size)
    self.assertEqual(larger_src_file_as_string, dst_key.get_contents_as_string())
    self.assertTrue(len(harness.transferred_seq_before_first_failure) > 1 and len(harness.transferred_seq_after_first_failure) > 1)