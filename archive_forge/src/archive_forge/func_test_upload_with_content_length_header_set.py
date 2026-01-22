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
def test_upload_with_content_length_header_set(self):
    """
        Tests resumable upload on a file when the user supplies a
        Content-Length header. This is used by gsutil, for example,
        to set the content length when gzipping a file.
        """
    res_upload_handler = ResumableUploadHandler()
    small_src_file_as_string, small_src_file = self.make_small_file()
    small_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    try:
        dst_key.set_contents_from_file(small_src_file, res_upload_handler=res_upload_handler, headers={'Content-Length': SMALL_KEY_SIZE})
        self.fail('Did not get expected ResumableUploadException')
    except ResumableUploadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
        self.assertNotEqual(e.message.find('Attempt to specify Content-Length header'), -1)