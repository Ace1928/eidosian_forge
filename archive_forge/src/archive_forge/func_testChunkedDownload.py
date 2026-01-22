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
def testChunkedDownload(self):
    bytes_http = object()
    http = object()
    download_stream = six.StringIO()
    download = transfer.Download.FromStream(download_stream, chunksize=26, total_size=52)
    download.bytes_http = bytes_http

    def _ReturnBytes(unused_http, http_request, *unused_args, **unused_kwds):
        url = http_request.url
        if url == 'https://part.one/':
            return http_wrapper.Response(info={'content-location': 'https://part.two/', 'content-range': 'bytes 0-25/52', 'status': http_client.PARTIAL_CONTENT}, content=string.ascii_lowercase, request_url='https://part.one/')
        elif url == 'https://part.two/':
            return http_wrapper.Response(info={'content-range': 'bytes 26-51/52', 'status': http_client.OK}, content=string.ascii_uppercase, request_url='https://part.two/')
        else:
            self.fail('Unknown URL requested: %s' % url)
    with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
        make_request.side_effect = _ReturnBytes
        request = http_wrapper.Request(url='https://part.one/')
        download.InitializeDownload(request, http=http)
        self.assertEqual(2, make_request.call_count)
        for call in make_request.call_args_list:
            self.assertRangeAndContentRangeCompatible(call[0][1], _ReturnBytes(*call[0]))
        download_stream.seek(0)
        self.assertEqual(string.ascii_lowercase + string.ascii_uppercase, download_stream.getvalue())