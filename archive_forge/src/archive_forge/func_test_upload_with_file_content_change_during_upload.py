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
def test_upload_with_file_content_change_during_upload(self):
    """
        Tests resumable upload on a file that changes one byte of content
        (so, size stays the same) while upload in progress.
        """

    def Execute():
        res_upload_handler = ResumableUploadHandler(num_retries=1)
        dst_key = self._MakeKey(set_contents=False)
        bucket_uri = storage_uri('gs://' + dst_key.bucket.name)
        dst_key_uri = bucket_uri.clone_replace_name(dst_key.name)
        try:
            dst_key.set_contents_from_file(test_file, cb=harness.call, res_upload_handler=res_upload_handler)
            return False
        except ResumableUploadException as e:
            self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
            test_file.seek(0, os.SEEK_END)
            self.assertEqual(test_file_size, test_file.tell())
            self.assertNotEqual(e.message.find("md5 signature doesn't match etag"), -1)
            try:
                dst_key_uri.get_key()
                self.fail('Did not get expected InvalidUriError')
            except InvalidUriError as e:
                pass
        return True
    test_file_size = 500 * 1024
    n_bytes = 300 * 1024
    delay = 0
    for attempt in range(2):
        test_file = self.build_input_file(test_file_size)[1]
        harness = CallbackTestHarness(fail_after_n_bytes=n_bytes, fp_to_change=test_file, fp_change_pos=1, delay_after_change=delay)
        if Execute():
            break
        if attempt == 0 and 0 in harness.transferred_seq_after_first_failure:
            delay = 15
            continue
        self.fail('Did not get expected ResumableUploadException')