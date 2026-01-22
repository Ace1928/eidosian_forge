from http import client as http_client
from oslotest import base as test_base
from ironicclient.common.apiclient import exceptions
def test_from_response_with_text_response_format(self):
    method = 'GET'
    url = '/fake-wsme'
    status_code = http_client.BAD_REQUEST
    text_data1 = 'error_message: fake message'
    ex = exceptions.from_response(FakeResponse(status_code=status_code, headers={'Content-Type': 'text/html'}, text=text_data1), method, url)
    self.assertIsInstance(ex, exceptions.BadRequest)
    self.assertEqual(text_data1, ex.details)
    self.assertEqual(method, ex.method)
    self.assertEqual(url, ex.url)
    self.assertEqual(status_code, ex.http_status)