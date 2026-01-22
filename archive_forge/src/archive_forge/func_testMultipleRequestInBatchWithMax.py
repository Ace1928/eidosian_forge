import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testMultipleRequestInBatchWithMax(self):
    mock_service = FakeService()
    desired_url = 'https://www.example.com'
    batch_api_request = batch.BatchApiRequest(batch_url=desired_url)
    number_of_requests = 10
    max_batch_size = 3
    for i in range(number_of_requests):
        batch_api_request.Add(mock_service, 'unused', None, {'desired_request': self._MakeSampleRequest(desired_url, 'Sample-{0}'.format(i))})
    responses = []
    for i in range(0, number_of_requests, max_batch_size):
        responses.append(self._MakeResponse(min(number_of_requests - i, max_batch_size)))
    with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as mock_request:
        self.__ConfigureMock(mock_request, expected_request=http_wrapper.Request(desired_url, 'POST', {'content-type': 'multipart/mixed; boundary="None"', 'content-length': 1142}, 'x' * 1142), response=responses)
        api_request_responses = batch_api_request.Execute(FakeHttp(), max_batch_size=max_batch_size)
    self.assertEqual(number_of_requests, len(api_request_responses))
    self.assertEqual(-(-number_of_requests // max_batch_size), mock_request.call_count)