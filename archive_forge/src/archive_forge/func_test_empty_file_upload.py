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
def test_empty_file_upload(self):
    """
        Tests uploading an empty file (exercises boundary conditions).
        """
    res_upload_handler = ResumableUploadHandler()
    empty_src_file = StringIO.StringIO('')
    empty_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    dst_key.set_contents_from_file(empty_src_file, res_upload_handler=res_upload_handler)
    self.assertEqual(0, dst_key.size)