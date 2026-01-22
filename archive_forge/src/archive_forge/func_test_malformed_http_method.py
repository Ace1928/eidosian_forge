import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_malformed_http_method(test_client):
    """Test non-uppercase HTTP method."""
    c = test_client.get_connection()
    c.putrequest('GeT', '/malformed_method_case')
    c.putheader('Content-Type', 'text/plain')
    c.endheaders()
    response = c.getresponse()
    actual_status = response.status
    assert actual_status == HTTP_BAD_REQUEST
    actual_resp_body = response.read(21)
    assert actual_resp_body == b'Malformed method name'
    c.close()