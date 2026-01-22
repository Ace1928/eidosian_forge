import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
def testAutoTransferCompressed(self):
    """Test that automatic transfers are compressed.

        Ensure uploads with the compressed, resumable, and automatic transfer
        flags set call StreamInChunks. StreamInChunks is tested in an earlier
        test.
        """
    upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, gzip_encoded=True)
    upload.strategy = transfer.RESUMABLE_UPLOAD
    with mock.patch.object(transfer.Upload, 'StreamInChunks') as mock_result, mock.patch.object(http_wrapper, 'MakeRequest') as make_request:
        mock_result.return_value = self.response
        make_request.return_value = self.response
        upload.InitializeUpload(self.request, 'http')
        self.assertTrue(mock_result.called)