import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testRefreshOnAuthFailure(self):
    mock_service = FakeService()
    desired_url = 'https://www.example.com'
    batch_api_request = batch.BatchApiRequest(batch_url=desired_url)
    desired_request = http_wrapper.Request(desired_url, 'POST', {'content-type': 'multipart/mixed; boundary="None"', 'content-length': 80}, 'x' * 80)
    with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as mock_request:
        self.__ConfigureMock(mock_request, http_wrapper.Request(desired_url, 'POST', {'content-type': 'multipart/mixed; boundary="None"', 'content-length': 419}, 'x' * 419), [http_wrapper.Response({'status': '200', 'content-type': 'multipart/mixed; boundary="boundary"'}, textwrap.dedent('                    --boundary\n                    content-type: text/plain\n                    content-id: <id+0>\n\n                    HTTP/1.1 401 UNAUTHORIZED\n                    Invalid grant\n\n                    --boundary--'), None), http_wrapper.Response({'status': '200', 'content-type': 'multipart/mixed; boundary="boundary"'}, textwrap.dedent('                    --boundary\n                    content-type: text/plain\n                    content-id: <id+0>\n\n                    HTTP/1.1 200 OK\n                    content\n                    --boundary--'), None)])
        batch_api_request.Add(mock_service, 'unused', None, {'desired_request': desired_request})
        credentials = FakeCredentials()
        api_request_responses = batch_api_request.Execute(FakeHttp(credentials=credentials), sleep_between_polls=0)
        self.assertEqual(1, len(api_request_responses))
        self.assertEqual(2, mock_request.call_count)
        self.assertEqual(1, credentials.num_refreshes)
        self.assertFalse(api_request_responses[0].is_error)
        response = api_request_responses[0].response
        self.assertEqual({'status': '200'}, response.info)
        self.assertEqual('content', response.content)
        self.assertEqual(desired_url, response.request_url)