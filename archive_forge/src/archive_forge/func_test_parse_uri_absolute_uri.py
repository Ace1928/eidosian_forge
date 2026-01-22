import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_parse_uri_absolute_uri(test_client):
    """Check that server responds with Bad Request to Absolute URI.

    Only proxy servers should allow this.
    """
    status_line, _, actual_resp_body = test_client.get('http://google.com/')
    actual_status = int(status_line[:3])
    assert actual_status == HTTP_BAD_REQUEST
    expected_body = b'Absolute URI not allowed if server is not a proxy.'
    assert actual_resp_body == expected_body