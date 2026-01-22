import datetime
from unittest import mock
import requests
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_from_response_overlimit(self):
    response = requests.Response()
    response.status_code = 413
    response.headers = {'Retry-After': '10'}
    body = {'keys': {}}
    ex = exceptions.from_response(response, body)
    self.assertEqual(10, ex.retry_after)
    self.assertIs(exceptions.OverLimit, type(ex))