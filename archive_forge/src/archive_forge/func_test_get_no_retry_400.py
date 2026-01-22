import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_get_no_retry_400(self):
    cl = get_authed_client(retries=0)
    self.requests = [bad_400_request, mock_request]

    def request(*args, **kwargs):
        next_request = self.requests.pop(0)
        return next_request(*args, **kwargs)

    @mock.patch.object(requests, 'request', request)
    @mock.patch('time.time', mock.Mock(return_value=1234))
    def test_get_call():
        resp, body = cl.get('/hi')
    self.assertRaises(exceptions.BadRequest, test_get_call)
    self.assertEqual([mock_request], self.requests)