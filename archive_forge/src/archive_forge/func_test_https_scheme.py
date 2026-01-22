import io
import urllib.error
import urllib.request
from oslo_config import cfg
import requests
from requests import exceptions
from heat.common import urlfetch
from heat.tests import common
def test_https_scheme(self):
    url = 'https://example.com/template'
    data = b'{ "foo": "bar" }'
    response = Response(data)
    mock_get = self.patchobject(requests, 'get')
    mock_get.return_value = response
    self.assertEqual(data, urlfetch.get(url))
    mock_get.assert_called_once_with(url, stream=True)