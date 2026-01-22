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
def test_upload_with_file_size_change_during_upload(self):
    """
        Tests resumable upload on a file that changes sizes while upload
        in progress
        """
    test_file_size = 500 * 1024
    test_file = self.build_input_file(test_file_size)[1]
    harness = CallbackTestHarness(fp_to_change=test_file, fp_change_pos=test_file_size)
    res_upload_handler = ResumableUploadHandler(num_retries=1)
    dst_key = self._MakeKey(set_contents=False)
    try:
        dst_key.set_contents_from_file(test_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.fail('Did not get expected ResumableUploadException')
    except ResumableUploadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
        self.assertNotEqual(e.message.find('File changed during upload'), -1)