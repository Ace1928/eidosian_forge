import socket
from unittest import mock
import io
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import testtools
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
from heatclient.tests.unit import fakes
@mock.patch('logging.Logger.debug', return_value=None)
def test_curl_log_i18n_headers(self, mock_log, mock_request):
    kwargs = {'headers': {'Key': b'foo\xe3\x8a\x8e'}}
    client = http.HTTPClient('http://somewhere')
    client.log_curl_request('GET', '', kwargs=kwargs)
    mock_log.assert_called_once_with(u"curl -g -i -X GET -H 'Key: foo㊎' http://somewhere")