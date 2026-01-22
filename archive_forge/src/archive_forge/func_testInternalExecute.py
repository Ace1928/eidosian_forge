import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testInternalExecute(self):
    with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as mock_request:
        self.__ConfigureMock(mock_request, http_wrapper.Request('https://www.example.com', 'POST', {'content-type': 'multipart/mixed; boundary="None"', 'content-length': 583}, 'x' * 583), http_wrapper.Response({'status': '200', 'content-type': 'multipart/mixed; boundary="boundary"'}, textwrap.dedent('                --boundary\n                content-type: text/plain\n                content-id: <id+2>\n\n                HTTP/1.1 200 OK\n                Second response\n\n                --boundary\n                content-type: text/plain\n                content-id: <id+1>\n\n                HTTP/1.1 401 UNAUTHORIZED\n                First response\n\n                --boundary--'), None))
        test_requests = {'1': batch.RequestResponseAndHandler(http_wrapper.Request(body='first'), None, None), '2': batch.RequestResponseAndHandler(http_wrapper.Request(body='second'), None, None)}
        batch_request = batch.BatchHttpRequest('https://www.example.com')
        batch_request._BatchHttpRequest__request_response_handlers = test_requests
        batch_request._Execute(FakeHttp())
        test_responses = batch_request._BatchHttpRequest__request_response_handlers
        self.assertEqual(http_client.UNAUTHORIZED, test_responses['1'].response.status_code)
        self.assertEqual(http_client.OK, test_responses['2'].response.status_code)
        self.assertIn('First response', test_responses['1'].response.content)
        self.assertIn('Second response', test_responses['2'].response.content)