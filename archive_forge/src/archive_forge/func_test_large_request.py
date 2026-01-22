import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
@pytest.mark.xfail(reason='https://github.com/cherrypy/cheroot/issues/106', strict=False)
def test_large_request(test_client_with_defaults):
    """Test GET query with maliciously large Content-Length."""
    c = test_client_with_defaults.get_connection()
    c.putrequest('GET', '/hello')
    c.putheader('Content-Length', str(2 ** 64))
    c.endheaders()
    response = c.getresponse()
    actual_status = response.status
    assert actual_status == HTTP_REQUEST_ENTITY_TOO_LARGE