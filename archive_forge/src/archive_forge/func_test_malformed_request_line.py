import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
@pytest.mark.parametrize(('request_line', 'status_code', 'expected_body'), ((b'GET /', HTTP_BAD_REQUEST, b'Malformed Request-Line'), (b'GET / HTTPS/1.1', HTTP_BAD_REQUEST, b'Malformed Request-Line: bad protocol'), (b'GET / HTTP/1', HTTP_BAD_REQUEST, b'Malformed Request-Line: bad version'), (b'GET / HTTP/2.15', HTTP_VERSION_NOT_SUPPORTED, b'Cannot fulfill request')))
def test_malformed_request_line(test_client, request_line, status_code, expected_body):
    """Test missing or invalid HTTP version in Request-Line."""
    c = test_client.get_connection()
    c._output(request_line)
    c._send_output()
    response = _get_http_response(c, method='GET')
    response.begin()
    assert response.status == status_code
    assert response.read(len(expected_body)) == expected_body
    c.close()